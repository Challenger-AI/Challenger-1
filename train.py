# alot upgraded version of karpathy/build-nanogpt, fp8 training, streaming dataset(no disk), larger tokenizer vocabulary etc..

import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd.function import Function
from huggingface_hub import HfApi
api = HfApi()

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

def _to_fp8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    x_f8  = (x * scale).clamp(finfo.min, finfo.max).to(dtype)
    return x_f8, scale.reciprocal().float()   # inverse for _scaled_mm

class _FP8Matmul(Function):
    @staticmethod
    def forward(ctx, x, w, out_dtype=torch.bfloat16):
        x_f8, x_inv = _to_fp8(x)
        w_f8, w_inv = _to_fp8(w)

        y = torch._scaled_mm(                     # row‑major A × col‑major B
            x_f8, w_f8.t(),
            out_dtype=out_dtype,
            scale_a=x_inv, scale_b=w_inv,
            use_fast_accum=True,
        )
        ctx.save_for_backward(x_f8, w_f8, x_inv, w_inv)
        ctx.out_dtype = out_dtype
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x_f8, w_f8, x_inv, w_inv = ctx.saved_tensors
        g_f8, g_inv = _to_fp8(grad_out, dtype=torch.float8_e5m2)

        # ---- dx = grad_out @  w ------------------------------------------
        # A  = g_f8                        (row‑major, (N, out))
        # B  = w_f8.T.contiguous().T       (col‑major, (out, in))
        dx = torch._scaled_mm(
            g_f8,
            w_f8.t().contiguous().t(),
            out_dtype=ctx.out_dtype,
            scale_a=g_inv, scale_b=w_inv,
            use_fast_accum=False,
        )

        # ---- dw = x.T @ grad_out  ----------------------------------------
        # A  = x_f8.T.contiguous()         (row‑major, (in, N))
        # B  = g_f8.T.contiguous().T       (col‑major, (N, out))
        dw = torch._scaled_mm(
            x_f8.t().contiguous(),
            g_f8.t().contiguous().t(),
            out_dtype=torch.float32,
            scale_a=x_inv, scale_b=g_inv,
            use_fast_accum=False,
        ).t()                                # bring back to (out, in)

        return dx, dw, None                  # no grad for out_dtype

# Convenience alias, identical signature to torch.mm
fp8_mm = _FP8Matmul.apply

# ---- drop‑in Linear ----------------------------------------------------------
class FP8Linear(torch.nn.Module):
    """Same signature as nn.Linear but weight‑stationary FP8 matmul."""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.trunc_normal_(self.weight, std=0.02)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Accepts x of shape (..., in_features) – any leading dims.
        Flattens to 2‑D, does the FP8 matmul, then restores the shape.
        """
        orig_shape   = x.shape[:-1]              # e.g. (B, T)
        x2d          = x.view(-1, x.shape[-1])   # (N, in_features)
        y2d          = fp8_mm(x2d, self.weight)  # (N, out_features)
        if self.bias is not None:
            y2d = y2d + self.bias
        y            = y2d.view(*orig_shape, self.weight.size(0))
        return y

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = FP8Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = FP8Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = FP8Linear(config.n_embd, 8 * config.n_embd)
        self.gelu    = nn.SiLU()
        self.c_proj  = FP8Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x, y = self.c_fc(x).split(x.size(-1) * 4, dim=2)
        x = self.gelu(x)
        x = self.c_proj(x * y)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        return x + self.attn(self.ln_1(x)) + self.mlp(self.ln_2(x))

@dataclass
class GPTConfig:
    vocab_size: int = 100288 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 16 # number of layers
    n_head: int = 16 # number of heads
    n_embd: int = 1536 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = FP8Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.NANOGPT_SCALE_INIT = 1

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, FP8Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x).float() # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-10, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
# -----------------------------------------------------------------------------
# NEW: streaming dataloader – put this in train_gpt2.py instead of the old
# DataLoaderLite implementation
# -----------------------------------------------------------------------------
from datasets import load_dataset

class DataLoaderLite:
    """
    Streams HuggingFaceFW/fineweb-edu and yields (x, y) tensors of shape (B, T),
    keeping at most one training batch worth of tokens in memory.

    - One dataloader instance is created per DDP rank.
    - The dataset is automatically sharded so each rank sees a disjoint subset.
    - Uses GPT‑2 tokenizer from `tiktoken`; every document is wrapped in an
      <|endoftext|> token.
    """

    def __init__(self, B, T, process_rank, num_processes, split):
        assert split in {"train", "val"}
        self.B, self.T = B, T
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.eot = self.enc._special_tokens["<|endoftext|>"]

        #
        # ---- Hugging Face streaming dataset ---------------------------------
        #
        # The `name=remote_name` argument is kept so that you still download the
        # 10 B‑token “sample‑10BT” variant.  Pull only the slice belonging to
        # this rank to avoid duplicated work in DDP.
        #
        self.dataset = (
            load_dataset(
                "hkust-nlp/PreSelect-100B",
                split=split,
                streaming=True,
            )
            .shard(num_shards=num_processes, index=process_rank)
        )

        # We keep an iterator and a local circular buffer of tokens.
        self.ds_iter = iter(self.dataset)
        self.buf: list[int] = []

    # -------------------------------------------------------------------------
    # public helpers -----------------------------------------------------------
    def reset(self):
        """Resets the iterator so that we start a new pass through the stream."""
        self.ds_iter = iter(self.dataset)
        self.buf.clear()

    def _fill_buffer(self, required: int):
        """
        Ensure the internal buffer has at least `required` tokens (+1 for the
        target shift).  Keeps pulling docs from the stream until the condition
        is met.  If the stream is exhausted (very unlikely in train split), we
        rewind and continue.
        """
        while len(self.buf) < required:
            try:
                doc = next(self.ds_iter)
            except StopIteration:           # end of stream → rewind
                self.reset()
                doc = next(self.ds_iter)

            # tokenise & append
            self.buf.append(self.eot)
            self.buf.extend(self.enc.encode_ordinary(doc["text"]))

    # -------------------------------------------------------------------------
    # main API -----------------------------------------------------------------
    def next_batch(self):
        """
        Returns: x, y tensors of shape (B, T) on CPU.
        The caller is responsible for .to(device) as before.
        """
        tokens_needed = self.B * self.T + 1  # +1 because we shift for targets
        self._fill_buffer(tokens_needed)

        # slice the first tokens_needed tokens, keep the remainder in buf
        tok = self.buf[:tokens_needed]
        del self.buf[:tokens_needed]

        tok = torch.tensor(tok, dtype=torch.long)         # (tokens_needed,)
        x = tok[:-1].view(self.B, self.T)                 # (B, T)
        y = tok[1:].view(self.B, self.T)                  # (B, T)
        return x, y

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("cl100k_base")

total_batch_size = 524288 * 2 # 2**20, ~1M, in number of tokens
B = 4 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=100288))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = True # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 2e-4 # 6b
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0:
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)
                api.upload_file(
                    path_or_fileobj=checkpoint_path,
                    path_in_repo="model.pt",
                    repo_id="MaxiiMin/Challenger-1",
                    repo_type="model",
                )

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
