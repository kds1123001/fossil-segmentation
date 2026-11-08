# fossilseg

A 3D U-Net that separates fossils from the surrounding rock in micro-CT scans. It is trained on synthetic data that I generate, because I don't have access to real scans or the months of hand-tracing needed to label them. I built it because I love dinosaurs.

This is a working pipeline, not a result. I have no numbers to report on real fossils yet, and the synthetic data makes the problem easier in a way I describe below.

## Why synthetic data

Segmenting fossils in micro-CT is hard because the fossil and the rock around it often have almost the same density. Studies of this problem describe very low contrast between fossil and matrix, and cases like shell in carbonate rock where density alone can't separate them (see the references). So a model has to rely on other cues, like texture and shape.

I can't get real scans, so `synth.py` builds volumes that fail the same way on purpose:

- Fossils are tubes, spirals, and blobs.
- The rock is a Perlin-style noise volume.
- The density distributions of fossil and rock overlap, so intensity alone doesn't give the answer.
- The local texture differs. The rock is grainier and the fossil is smoother. This is the signal the network is supposed to learn.
- Boundaries are blurred to imitate partial-volume effects. I also add ring artifacts, speckle noise, and a few cracks.

This is a stand-in. The fossils are simple shapes, and the texture difference is something I designed, not something I measured on real specimens.

## What's in the repo

```
fossilseg/
  synth.py       synthetic volume and mask generator
  model.py       3D U-Net, 4 downsamples, InstanceNorm, LeakyReLU
  losses.py      Dice + cross-entropy loss
  dataset.py     patch sampler with flip, rot90, and gamma augmentation
  train.py       training script
  infer.py       sliding-window inference with Gaussian-weighted blending
  mesh.py        marching cubes, exports .obj
  visualize.py   orthogonal slice viewer with mask and prediction overlay
scripts/
  generate_data.py
  predict.py
requirements.txt
```

## Running it

```
pip install -r requirements.txt

python scripts/generate_data.py --out_dir data/train --n 40 --shape 96 96 96 --seed0 0
python scripts/generate_data.py --out_dir data/val   --n 8  --shape 96 96 96 --seed0 10000

python -m fossilseg.train --train_dir data/train --val_dir data/val \
    --out_dir checkpoints --patch_size 64 --batch_size 2 --epochs 150 --steps_per_epoch 250

python scripts/predict.py --checkpoint checkpoints/best.pt --volume data/val/vol_0000.npy \
    --mask data/val/mask_0000.npy --out_dir predictions --mesh
```

Training and validation volumes use different seeds. `predict.py` saves the predicted mask as `.npy` and a PNG with three orthogonal slices and the prediction drawn on top. With `--mesh` it also writes an `.obj` file you can open in Blender or MeshLab.

## How it's built

**Model.** A plain 3D U-Net with a base width of 16 channels, which comes to about 6 million parameters. I used InstanceNorm instead of BatchNorm because batches of 3D volumes are tiny.

**Loss.** Dice plus cross-entropy, added together. Early in training the fossil class is only a couple of percent of the volume. Dice alone was unstable at that point, and cross-entropy alone didn't push hard enough on the small class.

**Patch sampling.** With only 1 to 3 percent of voxels being fossil, random crops are mostly pure rock, and the model learned to predict background everywhere. Biasing the crop centers toward fossil voxels fixed that in my runs. Some crops are still uniform, so the model sees background-only patches too.

**Inference.** Overlapping sliding windows with Gaussian blending at the seams. Straight tiling left visible block edges, and blending removed them.

## What is not shown

- **No results.** I have not reported a Dice or IoU score. The validation data comes from the same generator as the training data, so a good score there would not show the model works on anything else.
- **The model may be learning my shortcut.** Because I put the texture difference in on purpose, a simple local-variance filter followed by a threshold might do about as well as the U-Net. I haven't run that baseline.
- **No ablation.** I haven't tested what happens when fossil and rock have identical texture statistics. A model that really uses texture should fall apart there.
- **No test on real data.** Nothing here has seen a real scan.
- **The ring artifacts are crude.** They come from a simple radial sinusoid in `synth.py`, not from a proper simulation in sinogram space.

## Checks I want to run

- The local-variance baseline against the U-Net, on the same validation volumes.
- The identical-texture ablation.
- Evaluation on generator settings the network hasn't seen, such as different grain size and noise level.
- Pretraining on synthetic data and fine-tuning on a few labeled real slices, if I can find a public dataset.
- A SwinUNETR or UNETR model compared against this U-Net on the same splits, to see whether global attention helps on ambiguous boundaries.
- Proper sinogram-domain streak artifacts.
- Test-time augmentation with flips and rotations.
- Flagging high-uncertainty boundary voxels for a person to check.

## References

- Olaf Ronneberger, Philipp Fischer, and Thomas Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015).
- Ozgun Cicek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation" (2016).
- Fabian Isensee et al., "nnU-Net" (Nature Methods, 2021). My loss, patch sampling, Gaussian blending, and normalization choices follow its defaults.
- Espen Knutsen and Dmitry Konovalov, "Accelerating segmentation of fossil CT scans through Deep Learning" (Scientific Reports, 2024), doi:10.1038/s41598-024-71245-1.
- Stewart Edie et al., "High-throughput micro-CT scanning and deep learning segmentation workflow for analyses of shelly invertebrates and their fossils" (Frontiers in Ecology and Evolution, 2023), doi:10.3389/fevo.2023.1127756.

## Credits

Made by kds1123001 (aka dinokid64).

## License

MIT. See `LICENSE`.
