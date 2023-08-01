# :yum: CHANGELOG :yum:

This file describes the different changes performed at each update ! This will allow you to adapt your personnal code if something does not work anymore after an update.

Note that this file does not contain an exhaustive list but should at least contain the major updates / modifications in the signature of functions.

## Update 01/08/2023

### Main updates

- **[NEW](https://github.com/yui-mhcp/ocr)** Optical Character Recognition (OCR) project !
- Refactoring of the `utils.image.box_utils` module to a `directory` with much more features !
- **Experimental** support for long-text reading with the `Text-To-Speech (TTS)` models, which gives much better results for paragraph reading !
- Better `decorators` usage (especially the `timer` decorator, and datasets loadings) with the `functools.wraps` ! This allows to have a better `help` support for all decorated methods
- Some functions become numpy-friendly : instead of executing everything in pure tensorflow, some functions now adapt their internal functions to either call tensorflow, either numpy functions, depending on the input types. This feature is currently mainly supported in the `box_utils` module, but will be extended for performance critical operations !
```python
# Example : this function comes from `utils/distance/distance_method.py`, and simply expand dimensions of the input to perform matrix-based distance
# In this case, if the input is a `np.ndarray`, the function calls the `np.expand_dims`, avoiding a convertion to `tf.Tensor`, which may be unefficiant if not expected
# Note that this function is still graph-compatible ;)
def _maybe_expand_for_matrix(x, y, as_matrix = False):
    if isinstance(x, tf.Tensor):
        rank_fn, expand_fn = lambda t: len(tf.shape(t)), tf.expand_dims
    else:
        rank_fn, expand_fn = lambda t: t.ndim, np.expand_dims
    
    if as_matrix and rank_fn(x) == rank_fn(y):
        y = expand_fn(y, axis = -3)
    
    if rank_fn(x) == rank_fn(y) - 1:
        x = expand_fn(x, axis = -2)

    return x, y

```

## Image features

- Modification of the `BaseImageModel` to support image resizing configuration in a more flexible way (cf `resize_config` params + modification in the `get_image` method for variable input size support)
- Refactoring of the `box_utils` module, especially the `get_box_pos` (which supported 3 box formats) has been replaced by the `convert_box_format` that handles more `BoxFormat`, and now supports the `tensorflow graph` mode !
- New padding methods supported (used in the `OCR` project)
- Implementation of the `Locality Aware NMS` for the `EAST` Scene-Text Detection model

## Text features

- The `text_decoder` module has been removed (because it was never used), and replaced by the `ctc_decoder` module, giving a better interface to `CTC`-decoding. Consequently, the `TextEncoder` has a new `ctc_decode` function

## Update 01/06/2023

### Main updates

**IMPORTANT NOTE** : the `requirements.txt` have been cleaned up in all the projects to remove some *heavy but rarely used* dependancies. It is therefore possible that some specific functions raise errors (especially for model convertions that may require `pytorch` or `transformers`). Nevertheless, this may not be an issue in most of the usages, and is easily solvable by `pip install` it ! ;)

- The `EAST Scene-Text Detector` has been modified to support the version of [this repo](https://github.com/SakuraRiven/EAST), which has powerful pretrained model published ! The training procedure is however not supported yet (but not necessary to be used).
- The [Speech-To-Text Whisper](https://github.com/yui-mhcp/speech_to_text) has been refactored to be equivalent to [the official OpenAI's repo](https://github.com/openai/whisper) ! It supports all the multilingual versions (i.e. base, medium, large, ...). The ".en" versions have not been fully tested yet (especially the tokenizer).
- The `Transformers` architectures now correctly support the `mixed_precision` or `float16` !
- The coding style of some models is improved to better support the methods from the interfaces. The major benefit is that the code is more clear, simpler, and the docstrings / signatures from `get_image` are properly shown when using `help(model.get_input)`, which was not the case before.
```python
class ImageModel(BaseImageModel):
    # new style
    get_input = BaseImageModel.get_image
    
    # old style
    def get_input(self, data, ** kwargs):
        return self.get_image(data, ** kwargs)
```

## Image features

- New `rotate_image` method, to rotate the image in pure tensorflow (the code is highly inspired from the `RandomRotate` layer). 
- New `random_rotate` and `random_resize` image augmentation methods
- New `get_image_augmentation_config` method that returns all supported kwargs (with default values) for image augmentation methods (useful to track these default configuration in the training hparams).
- Support for `bbox` kwarg in the `load_image` method (in graph mode).
- New `get_image_with_box` method in the `BaseImageModel`, that can use the `bbox` argument of `load_image` if the data contains a `box` key. This is not done by default as the `YOLO` model aims to detect the box, so the `box` field is for the output and has to be ignored for the input processing.
- New support for `target_max_shape` and `target_multiple_shape` in `resize_image` and `load_image` methods (see their documentation for details). 
- New `{down / up}sampling_factor` in the `BaseImageModel`, allowing to get information on the downsampling / upsampling facotrs of the model (useful to determine the `target_multiple_shape` for UNet-like models with variable input size). Check the `EAST` model for an example usage.

## Update 01/05/2023

### Main updates

**WARNING** Some projects have been refactored, and the signatures of the prediction methods may have changed. It may be useful to check them if any error occurs in your custom code ;) The remaining projects will be refactored in the future updates, it is therefore possible that some features are temporaly not supported.

- (**Experimental**) New model for Scene-Text Detection : [EAST](https://github.com/yui-mhcp/detection) (+ `U-Net` architecture and `Dice` loss + rotated bounding boxes).
- Refactoring / optimization of the [Text-To-Speech](https://github.com/yui-mhcp/text_to_speech) prediction pipeline + simplification of the `Vocoder`.
- Refactoring / optimization of the [object detection](https://github.com/yui-mhcp/detection) prediction pipeline.
- Refactoring / optimization of the [image captioning](https://github.com/yui-mhcp/image_captioning) prediction pipeline.
- Better API for the [image captioning](https://github.com/yui-mhcp/image_captioning) via the new `make_caption` method.
- Refactoring / cleaning of the [Siamese Networks](https://github.com/yui-mhcp/siamese_networks) main class.
- Support for 3D plots.

## Bugs fixed

- [TTS project](https://github.com/yui-mhcp/text_to_speech) : the `get_audio_file` is now able to get the audio file associated to a non-cleaned text, by passing the model used to generate it. The function cleans the text by loading the model's `TextEncoder`, and finally matches the cleaned text.

## Distance features

- The `knn` function and `KNN` class supports string ids.
- The `tf_distance` now supports the `max_matrix_size` in `tensorflow graph` mode.

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