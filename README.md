# microgpt visualized

An interactive, single-file visualization of [microgpt.py by Andrej Karpathy](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95): a complete GPT (autograd, attention, MLP, Adam, sampling) in about 200 lines of dependency-free Python.

The page runs the real algorithm in your browser and lets you watch it one operation at a time.

## Run it

Open `index.html` in a browser. There is no build step and no server.

To host it, enable GitHub Pages for this repository (Settings → Pages → Deploy from a branch → `main`, `/root`).

## What you can do

- **Play, pause, step forward, step back, scrub.** One training step is broken into frames: tokenize the name, then for every position embed → query/key/value → attention weights → residual add → MLP → logits → softmax and loss, then the mean loss, the backward pass and the Adam update. Speed goes from 0.5× to 8×. Space toggles play; the arrow keys step.
- **Follow the code.** The lines of `microgpt.py` that produce the current frame are highlighted in the listing.
- **Read every number.** Each square is one scalar from the model. Hover it for the exact value, and for its gradient once the backward pass has run.
- **See gradients.** On the backward frame the diagram switches from values to gradients; click any token to inspect its position.
- **Train fast.** Finish the current step, run 100 steps, or train to the end without animation while the loss curve and all 4,192 parameters update live.
- **Generate.** Switch to Generate to watch the model sample a name token by token. Change the temperature, or type the first letters yourself.
- **Change the experiment.** Paste your own word list, change the learning rate, the number of steps or the seed, and rebuild the model.

## How faithful is it

The engine inside `index.html` is a JavaScript port of `microgpt.py` with the same architecture and hyperparameters (1 layer, 16-dim embeddings, 4 heads, block size 16, Adam with linear learning-rate decay). Two things differ, and neither changes the maths:

- A dot product is a single autograd node instead of a chain of multiply and add nodes, and the topological sort is iterative. This is what makes it fast enough to train in a browser. Gradients were checked against finite differences.
- JavaScript's random numbers are not Python's, so values will not match a Python run with seed 42. The built-in dataset is about 1,300 first names embedded in the page rather than the 32,000-name `names.txt`, because the page works offline.

## Files

- `index.html` — the whole visualization
- `microgpt.py` — the original program, for reference

## Credit

`microgpt.py` is by Andrej Karpathy. This repository only adds the visualization.
