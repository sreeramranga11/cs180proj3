# CS 180 Project 3 – (Auto)stitching and Photo Mosaics

## Reproducing The Results

1. **Install dependencies**  
   ```bash
   pip install numpy scipy imageio matplotlib pillow
   ```

3. **Run the deliverable generator**  
   Regenerate every artifact (correspondences, warped images, mosaics, rectifications) from the raw photos in `images/`:
   ```bash
   python generate_deliverables.py \
     --images-dir images \
     --output-dir deliverables
   ```

   - Add `--datasets bedroom stairway` to process a subset.  
   - Add `--skip-mosaics` or `--skip-rectifications` to limit the stages.  
   - Pass `--reference-index` to choose a different middle image.

4. **View the report**  
   Open `index.html` in a browser; it will pick up the refreshed outputs from `deliverables/`.

> **Note:** The complete deliverables directory exceeds 1 GB. I had to remove the debug subdirectory to publish when deploying to GitHub Pages.