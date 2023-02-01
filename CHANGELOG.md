# :yum: CHANGELOG :yum:

This file describes the different changes performed at each update ! This will allow you to adapt your personnal code if something does not work anymore after an update.

Note that this file does not contain an exhaustive list but should at least contain the major updates / modifications in the signature of functions.


## Update 01/02/2023

### Major updates

- New CHANGELOG file ! :smile:
- New [STT](https://github.com/yui-mhcp/speech_to_text) model : [Whisper](https://github.com/openai/whisper) !
- New dataset's information tracking in the `train` and `test` functions ! The `History` class now keeps track of `optimizer / loss / metrics` configuration as well as (optionally) a summary of the dataset's columns (check the `datasets.summarize_dataset` function)
- New `additional_infos` kwarg to `train` and `test` functions to manually add some information to keep in the `History.trainings_config` of this training round

### Bugs fixed

- Minor bug in `prepare_dataset` : if `batch_before_map` was set to `True`, the `cache` method was called 2 times (I do not know if it has an impact or not but probably not a major one)
- Some shared models have been updated :
    - `yolo_faces`      : due to a `Bad marshall data` when loading the model
    - `sv2tts_siwis_v2` : the default embeddings have been changed to a `.csv` file (instead of the `.pdpkl` one) due to an error in the `Colab` demo
    - All the `CLIP` models should be re-created as their signature has changed
- Update of `masking_1d` layers to support the `dilation_rate` argument when masking

### General features

- The `unitest` framework has been updated to better support some features raising some errors when saving
- The default `embeddings`' extension is now `.csv` (instead of `.pdpkl`)

### Text features

- The `BaseTextModel.text_signature` ahs changed from a tuple `(token, length)` to `tokens` (i.e. the `length` has been removed from all text's signatures)

The reason is that text padding is faster with `tf.mat.not_equal` compared to `tf.sequence_mask`. Therefore, the input's length has been replaced by the `set_tokens` / `pad_token` kwarg for text-based models (mainly `Transformers` subclasses). 

- All the methods from `BaseTextModel`'s subclasses in the processing pipeline (`encode_data`, `filter_data`, `augment_data` and `preprocess_data`) have changed to not take / return the input length.
- Furthermore, the input sequences do not have the `eos_token` anymore but are always finished by `pad_token`.
    - Before : the input sequence was `[SOS] *tokens [EOS] *[PAD]` and the `input_length` was equal to `1 + len(tokens)` (so `[EOS]` was not taken into account as input).
    - Now    : the input sequence is `[SOS] *tokens *[PAD]` without `input_length`

**Transformers still support the `input_length` argument but it is not used anymore by default**

### Image features

- Support for the `COCO` polygon-based segmentation (`create_poly_mask`)
- The `create_color_mask` supports the `tensorflow graph` mode
- The `_normalize_color` has changed to `normalize_color` and has a `tf-graph` version `tf_normalize_color`

### Audio features

- New `read_ffmpeg` function to read audio from audio / video files. It has been set as default for video files
- The `read_{}` functions may now get additional arguments so they should have a `** kwargs` in their signature. This is for instance used in the `read_ffmpeg(filename, rate = None, ** kwargs)` as `ffmpeg` can support resampling while loading the audio