# DEDNet
## Architecture
<img width="1461" height="635" alt="image" src="https://github.com/user-attachments/assets/e6af41df-2728-4c16-89bf-32aef6f1664a" />
The proposed DEDNet network comprises four main components: Feature Extractor,
Audio-Visual Modality Encoder, Relationship Subgraph Interaction Module, and Emotion Classifier. Firstly, three distinct
feature extractors are employed to obtain the initial features
at the utterance level. Subsequently, for textual modality
information, we design the Relationship Interaction Subgraph
Module to acquire features containing inter-speaker emotional dependency and intra-speaker emotional dependency.
Regarding the Audio-Visual Encoder, we utilize a Transformer
along with speaker embedding to obtain modality features
incorporating speaker information. Finally, by integrating the
fused multimodal features with each unimodal feature, we
construct an Emotion Classifier and Auxiliary Fusion Loss
Function to achieve multimodal emotion classification.****
