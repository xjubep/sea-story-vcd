# Video Copy Detection with Sea Story
### 1. Decode video (frame, audio, audio segment)
   * `decode_video.py`
### 2. Extract frame feature and get segment feature (vision)
   * `extract_feature.py`
   * `get_segment_feature.py`
### 3. Extract audio segment feature (MFCC) 
   * `extract_audio_feature.py`
### 4. Get multi modal feature
   * `get_multi_feat.py`
### 5. Analyze result and get f1 score
   * `analyze.py`: visualize feature using PCA
   * `get_f1.py`: search the best temporal network hyperparameter by f1 score
   * `copy_detection.py`: get tiled image of key frames and videos for suspected copy segment
