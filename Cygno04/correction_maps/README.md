# Correction maps to be applied to remove the OLD vignetting to CYGNO04 reco data

The currently (October 2026) reconstructed data of CYGNO04 use an old vignetting. These maps, done for many combinations of VGEM and Vdrift:

1. De-correct the old vignetting
2. Correct for the new one and possibly other [x-y] disuniformities

this is done simply from reconstructed data without source taken at LNF with 50ms exposure, assuming a uniform illumination in [x-y] once integrating over many images.

These maps can be visualized [here](https://emanuele.web.cern.ch/emanuele/Cygnus/plots/cygno04/commissioning_lnf/?match=prof2d_cygno04_sideA_muons_withVignetting_VGEM_*_VD_*). A presentation at the reconstruction meeting is [here](https://agenda.infn.it/event/54202/#6-uniformity-maps-cygno-04).

## Usage: 
- The maps are TH2F with 1 bin = 16 x 16 pixels.
- This is LY profile, correction should be 1/bincontent
