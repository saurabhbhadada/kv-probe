## The easiest way to think about the whole project is:

> We are asking whether **keys inside a transformer's KV cache have a hidden hierarchical/binary structure**, and whether that structure tells us something useful about how the model will use those keys later.

If yes, that structure might eventually let us compress the KV cache more intelligently than ordinary quantization.

### Start with one token

Suppose the model reads:

> `The cat sat on the mat`

For every token, each attention head creates a **Key** vector.

Very simplified:

```text
"The" → K₁ = [ ... ]
"cat" → K₂ = [ ... ]
"sat" → K₃ = [ ... ]
"on"  → K₄ = [ ... ]
```

A real key might have 256 numbers:

\[
K_{\text{cat}}
=
[0.14,-0.72,0.03,\ldots]
\]

When future tokens arrive, their Queries compare themselves against these Keys.

So later, if the model generates:

> `because it was tired`

the queries for `"because"`, `"it"`, `"was"`, `"tired"` may repeatedly attend to `"cat"`.

That is why keys matter.

---

## What we're trying to discover

Suppose we take two old keys:

\[
K_i,\;K_j
\]

Maybe they came from `"cat"` and `"dog"`.

We ask:

> If these two keys are structurally similar, do future tokens also treat them similarly?

This gives us two sides of the experiment:

```text
How similar are Kᵢ and Kⱼ?
             ↓
      our candidate metric

How similarly are they used later?
             ↓
       attention behavior
```

Then we test whether the first predicts the second.

---

# Ordinary similarity first

The obvious baseline is **cosine similarity**.

If:

```text
Kcat = [0.8, 0.4]
Kdog = [0.7, 0.5]
```

they point in almost the same direction.

So cosine might be:

\[
0.98
\]

meaning:

> very similar.

But we want to test whether **p-adic structure** might reveal something cosine misses.

---

# Where p-adics enter

Neural-network keys are floating-point numbers.

For the p-adic experiment, we first convert them into integers using a fixed scale.

For example:

```text
floating key:

[0.12, 0.34, -0.18]

× fixed scale 100

integer key:

[12, 34, -18]
```

The important change we're making now is:

> Every key inside the same attention head uses the **same scale**.

So we have one consistent integer space.

That's necessary because otherwise comparing p-adic distances wouldn't mean much.

---

# What does \(v_2\) mean?

For an integer \(x\),

\[
v_2(x)
\]

asks:

> How many times can I divide this number by 2?

Examples:

\[
12 = 2^2\times3
\]

so:

\[
v_2(12)=2.
\]

And:

\[
8=2^3
\]

so:

\[
v_2(8)=3.
\]

While:

\[
7
\]

is odd, so:

\[
v_2(7)=0.
\]

---

# How do we compare two numbers?

Suppose two key coordinates are:

\[
12,\;20
\]

Their difference is:

\[
20-12=8.
\]

Because:

\[
8=2^3
\]

we get:

\[
v_2(20-12)=3.
\]

That says those two integers share **three levels of 2-adic similarity**.

But take:

\[
12,\;19.
\]

Difference:

\[
19-12=7
\]

and:

\[
v_2(7)=0.
\]

So they are not close in the 2-adic sense.

This is very different from ordinary distance.

---

# Now apply that to whole Key vectors

Imagine:

```text
K₁ = [12, 20, 36, 7]
K₂ = [20, 28, 40, 9]
```

Coordinate differences are:

```text
8, 8, 4, 2
```

The valuations are:

```text
v₂(8) = 3
v₂(8) = 3
v₂(4) = 2
v₂(2) = 1
```

So:

```text
[3, 3, 2, 1]
```

Now define something like **S2**:

> What fraction of dimensions have valuation at least 2?

Here:

```text
3 ≥ 2 ✓
3 ≥ 2 ✓
2 ≥ 2 ✓
1 ≥ 2 ✗
```

so:

\[
S_2=\frac34=0.75.
\]

Meaning:

> 75% of these coordinates share at least two levels of binary/p-adic structure.

That's currently one of our most important quantities.

Similarly:

\[
S_1,S_2,S_3,S_4
\]

ask progressively stricter versions of the same question.

---

# What are we comparing S2 against?

Now comes the clever part.

We don't just ask whether two keys are mathematically similar.

We ask whether the model **uses them similarly later**.

Imagine keys belonging to token positions:

```text
position 10 → K₁
position 15 → K₂
```

Later queries might attend to them like:

```text
future token        attention to K₁    attention to K₂

token 20               0.20               0.22
token 21               0.40               0.38
token 22               0.10               0.12
token 23               0.31               0.30
```

Those two columns are extremely similar.

So we'd say:

> K₁ and K₂ have similar **future attention behavior**.

Contrast that with:

```text
future token        attention to K₁    attention to K₂

token 20               0.80               0.02
token 21               0.05               0.60
token 22               0.70               0.01
token 23               0.01               0.70
```

Those keys behave very differently.

This is why we fixed the old row/column bug.

We're interested in:

\[
A_{\text{future},i}
\]

and:

\[
A_{\text{future},j}
\]

because they tell us:

> How do future queries treat key \(i\) versus key \(j\)?

---

# So every sampled pair gives us something like this

For two keys:

```text
Kᵢ and Kⱼ
```

we compute:

```text
cosine similarity       = 0.72
Euclidean similarity    = ...
S1                      = 0.81
S2                      = 0.57
S3                      = 0.29
future attention sim.   = 0.76
```

Then we do this for thousands of key pairs.

Now we have data like:

| pair | S2 | cosine | future attention similarity |
|---|---:|---:|---:|
| A | 0.75 | 0.62 | 0.81 |
| B | 0.12 | 0.55 | 0.20 |
| C | 0.68 | 0.90 | 0.73 |
| D | 0.03 | 0.15 | 0.05 |

And we ask:

> Which metric predicts future behavior better?

---

# That's where Spearman correlation comes in

Suppose across thousands of pairs:

\[
\rho_{S2}=0.42
\]

and:

\[
\rho_{\cos}=0.25.
\]

Then:

\[
\Delta\rho
=
0.42-0.25
=
0.17.
\]

That would mean:

> S2 ordering tracks future attention behavior better than cosine similarity does.

That's potentially very interesting.

Our main number is:

\[
\boxed{
\Delta\rho
=
\rho(S_2,\text{future attention similarity})
-
\rho(\text{cosine},\text{future attention similarity})
}
\]

If that number is consistently positive, p-adic structure may contain useful information.

---

# Why do we analyze each head separately?

An attention head is basically its own little specialist.

One head might track:

```text
syntax
```

another:

```text
long-range references
```

another:

```text
position
```

etc.

So combining all heads could hide a signal.

We instead want:

```text
Head 0 → Δρ = +0.01
Head 1 → Δρ = +0.18
Head 2 → Δρ = -0.03
Head 3 → Δρ = +0.21
...
```

If heads 1 and 3 strongly favor p-adic structure, that may itself tell us something interesting.

Maybe only certain transformer computations naturally create hierarchical representations.

---

# Why the permutation control matters

Suppose we get:

\[
\rho_{S2}=0.40.
\]

Great?

Not necessarily.

Maybe FP16 or binary quantization creates some accidental p-adic pattern.

So we take the real keys and **shuffle which token position they belong to**.

For example:

```text
original:

position 1 → K1
position 2 → K2
position 3 → K3
position 4 → K4
```

becomes:

```text
position 1 → K3
position 2 → K1
position 3 → K4
position 4 → K2
```

Same numbers.

Same FP16 representation.

Same distribution.

But the learned connection between:

```text
key ↔ token ↔ attention behavior
```

has been destroyed.

If:

```text
real       rho_S2 = 0.40
permuted   rho_S2 = 0.02
```

that's very encouraging.

If instead:

```text
real       = 0.40
permuted   = 0.37
```

then we probably discovered a numerical artifact rather than transformer structure.

---

# Why bootstrap?

Imagine we get:

\[
\Delta\rho=+0.08.
\]

Is that real, or did we just get lucky with our sampled text?

Bootstrap estimates uncertainty.

Conceptually:

```text
Run 1 → +0.07
Run 2 → +0.09
Run 3 → +0.06
Run 4 → +0.10
...
```

Then maybe:

\[
95\%\ CI=[+0.04,+0.11].
\]

Since zero is outside that range, that's evidence the advantage is stable.

But:

\[
95\%\ CI=[-0.03,+0.14]
\]

would mean:

> We can't yet tell whether p-adic similarity is actually better.

And we're bootstrapping by **sequence**, because thousands of pairs coming from the same text aren't independent observations.

---

# And why p=3 and p=5?

Our main idea currently uses:

\[
p=2.
\]

But we also test:

\[
p=3,\quad p=5.
\]

These are diagnostics.

Suppose:

```text
p=2 → strong signal
p=3 → weak
p=5 → weak
```

That might suggest something specifically related to binary structure.

But if:

```text
p=2 → strong
p=3 → strong
p=5 → strong
```

maybe what we're seeing is a more general modular/hierarchical phenomenon rather than specifically 2-adic structure.

We don't know yet.

That's why we're probing.

---

# If Experiment 1 works, what happens next?

Suppose we eventually find:

```text
Head 3:

S2 correlation       = 0.46
cosine correlation   = 0.25

Δρ                   = +0.21
95% CI               = [+0.15,+0.26]

permuted S2          = 0.04
```

That would be a **very interesting result**.

It would suggest:

> There is hidden p-adic congruence structure in transformer keys that contains information about future attention behavior that ordinary cosine geometry does not capture.

Then we move from:

### "Does the structure exist?"

to:

### "Can we exploit it?"

For example:

```text
KV cache
    ↓
group p-adically similar keys
    ↓
build hierarchy/tree
    ↓
merge/compress old related keys
    ↓
retain important representatives
```

And that becomes **PadicKV**.

So right now we're not really trying to compress anything yet.

We're doing the more fundamental experiment:

\[
\boxed{
\text{Is there actually something there worth exploiting?}
}
\]

If Experiment 1 says **no**, that's useful—we stop before spending months building a compression algorithm around a false premise.

If it says **yes**, then the project becomes much more exciting.
