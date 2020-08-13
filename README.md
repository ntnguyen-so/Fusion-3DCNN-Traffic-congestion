# The github repository storing source code of the Fusion-3DCNN deep learning model.

### The solution is accepted to be published at The 31st International Conference on Database and Expert Systems Applications - DEXA2020
Camera-ready version of the paper can be retrieved at: https://github.com/thanhnn-uit-13/Fusion-3DCNN-Traffic-congestion/blob/master/Final_paper.pdf.

### Complete workflow ###
![Complete workflow](_imgs/workflow.png?raw=true)

### Architecture of Fusion-3DCNN ###
![Fusion-3DCNN](_imgs/architecture.png?raw=true)

### Sample predictions of Fusion-3DCNN ###
* Fusion-3DCNN which uses (1) traffic congestion, (2) precipitation, (3) vehicle collisions, and (4) Tweets (SNS data) about the first 3 factors for short-term dataset. The model looks back urban sensing data for 3 hours to predict traffic congestion for 1.5 hours.
![short](_imgs/sample_short_predicted.png?raw=true)

* Fusion-3DCNN which uses (1) traffic congestion, (2) precipitation, (3) vehicle collisions, and (4) Tweets (SNS data) about the first 3 factors for long-term dataset. The model looks back urban sensing data for 1 day to predict traffic congestion for 1 day.
![long](_imgs/sample_long_predicted.png?raw=true)

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

