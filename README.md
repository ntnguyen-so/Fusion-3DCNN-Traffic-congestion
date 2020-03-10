# The github repository storing source code presented in "Fusing Multi-source Urban Sensing Data for Raster-Image-based 3D-CNN in Traffic Congestion Prediction" paper.

### Repository structure ###
* timeseries2raster: the algorithm to convert sensing data stored in time-series format to 2D Multi-layer Raster-Images.
* prep3Draster: the algorithm to convert 2D Multi-layer Raster-Images to 3D Multi-layer Raster-Images.
* learning_model: 
    * Fusion-3DCNN_CPA: Fusion-3DCNN which uses (1) traffic congestion, (2) precipitation, and (3) vehicle collisions.
    * Fusion-3DCNN_CPA_SNS: Fusion-3DCNN which uses (1) traffic congestion, (2) precipitation, (3) vehicle collisions, and (4) Tweets (SNS data) about them.
    * baselines:
         * Historical_Average: Historical Average model.
         * Seq2Seq: Sequence-to-sequence Long Short-term Memory model.
         * 2D-CNN: 2D-CNN model
