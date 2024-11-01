# :yum: CHANGELOG :yum:

This file describes the different changes performed at each update ! This will allow you to adapt your personnal code if something does not work anymore after an update.

Note that this file does not contain an exhaustive list but should at least contain the major updates / modifications in the signature of functions.

## Update 01/11/2024

### Major update

- **[NEW]** [Installation guide](INSTALLATION.md) that provides step-by-step instructions to install `tensorflow` and `TensorRT-LLM` in a `mamba` virtual environment, along with all the expected additional softwares !
- **[NEW]** The `BaseModel` class now supports different build modes with `tf.saved_model` and `TensorRT-LLM` models (in addition to the regular `keras` model) ! This feature is experimental, and may have some limitations ;)
- **[NEW]** A new project on [Language Models](https://github.com/yui-mhcp/language_models) is released with a first set of `Natural Language Understanding (NLU)` tasks, such as Machine Translation, Summarization, Text Reformulation, and more ! :yum:
- A new [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) runtime wrapper is proposed in the new [Language Models](https://github.com/yui-mhcp/language_models) project ! This enables a highly optimized inference with most of the state-of-the art `Large Language Models (LLM)` ! :yum:
- **[BREAKING CHANGE]** The `graph_compile` decorator now returns an object (`CompiledFunction`) that offers additional features, such as the `export` method to save it into a `tf.saved_model` format. This feature is experimental and may have some limitations ;)
- **[EXPERIMENTAL]** A new `utils/search/web` module is released allowing to perform web-based search with multiple search engines ! However, some of them require additional tokens, so I was not able to test them all. 

### Known issues

- The `utils.clustering` module has some errors when executed in graph/XLA mode. 
- The `WaveGlow` keras checkpoint is missing in the [Text-to-Speech](https://github.com/yui-mhcp/text_to_speech) and will be added in the next update
- The `models.tts.stream` method is not working anymore since keras 3 update. It will be solved in the next update

### General features

- The `utils.keras_utils.ops` module is now version-independant and should be compatible with any version of `keras>3.0`
- The `utils.keras_utils.compile` module is now a directory with additional features to load/export compiled functions to `tf.saved_model`
- A new `utils.parser` file has been created to include all functions related to function inspection (these features were initially in `utils.generic_utils`)
- `pandas` is now an optional dependency
- **[BREAKING CHANGE]** The `{load / save}_embeddings` have been optimized with modifications in their name/signature

### Text features

- The `TextEncoder.format` and `TextEncoder.apply_template` both supports `f-string` and `jinja` formats
- The `document_parser` module has been unified such that all parsing methods return similar structure
- 


## Update 15/07/2024

### Major update

- The [detection](https://github.com/yui-mhcp/detection) and [ocr](https://github.com/yui-mhcp/ocr) prediction and streaming methods have been cleaned up and optimized ! :smile:
    - They now use the new `utils/callbacks` module that abstracts some common file saving / result display strategies
    - They are based on the new `models/utils/prediction.py` module that abstracts data preparation for prediction. This has enabled some useful features such as directory / formatted files support for both `predict` methods
    - The `stream` methods support fine-grained configuration to save raw stream, transformed stream, frames, ...

This new abstraction is still experimental, and will be generalized to all other model classes in the near future !

### Bugs fixed

- The `Beam Search inference` is fixed (minor issue for long generations)
- The `BaseOCR.stream` is now properly working
- The `BaseDetector.predict` now correctly supports directories / formatted files
- The `video_utils.copy_audio` now supports multiple calls in parallel
- The `image_io.stream_camera` now correctly saves videos when the directory did not exist yet

### Known issues

- The `WaveGlow` keras checkpoint is missing in the [Text-to-Speech](https://github.com/yui-mhcp/text_to_speech) and will be added in the next update
- The `models.tts.stream` method is not working anymore since keras 3 update. It will be solved in the next update

## Update 01/07/2024

### Major update

- The `utils/keras_utils/ops` has been refactored to support **all** the `keras.ops` functions, and even some additional custom functions !
- A `TextEncoder` model has been added for information retrieval in the [encoders](https://github.com/yui-mhcp/encoders) module !
- The `utils/search` module has been created for [the new information retrieval](https://github.com/yui-mhcp/encoders) feature ! :smile:
- The `euclidian_distance` and `dot_product` have been optimized using the `einsum` function ! This has enabled a **huge** memory saving and performance increase. If you are interested in the benchmarks, please open a discussion ! :smile:
- The `get_pretrained_transformer` method automatically infers the class based on the `transformers` model, and downloads metadata file instead of instanciating the model
- The `Transformers.from_pretrained` method has been abstracted to correctly build config / transfer weights for all the supported architectures !

### Known issues

- The `spectral_clustering` and `label_propagation` clustering methods seem to not work properly anymore.
- The `spectral_clustering` is not working at all in `tf2.17.0`, due to an error with the `svd` method
- The `pypdf`-based parsing method is not properly working, as some information on the text position are not provided by the library. It is recommanded to **not** use it currently

### Bugs fixed

- The `YOLO` class now correctly forwards the labels to the detected boxes
- The `draw_boxes` and `show_boxes` now correctly displays the label
- The `plot_embeddings` now properly supports `Tensor` ids
- Fix minor bug in `SentencepieceEncoder` decoding method (the `offset` was not properly saved, which only impacted the `decode` method)

### General features

- All the `requirements` files have been updated
- A new experimental `apply_on_batch` decorator has been created to abstract functions that iterate over batches
- New `segment_*` operations support, allowing optimization in some functions, such as the `compute_centroids` method
- The `{save / load}_embeddings` have been optimized, and the default embedding format is now `.h5`
- The `.h5` file format is now properly supported to store nested `dict`

### Image features

- The `draw_boxes` method now displays the label over a background rectangle, enhancing readibility

### Text features

- The `pdf_parser` has been refactored to support multiple pdf extraction libraries, and the default has been switched to [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) instead of [pdfminer.six](https://github.com/pdfminer/pdfminer.six), allowing a significant performance speed up
- A new post-processing strategy has been designed for the `pypdfium2`-based pdf extraction, and will be generalized to `pypdf` and `pdfminer` parsers in the future
- A new `pd_parser` has been added, designed to parse `.md` files
- The `parse_document` method now accepts directories / file formats


## Update 01/06/2024

The [data processing](https://github.com/yui-mhcp/data_processing), [base_dl_project](https://github.com/yui-mhcp/base_dl_project) and [detection](https://github.com/yui-mhcp/detection) projects have been updated.

### Major updates

- The `BaseModel` class has been completely refactored (as well as the other `interfaces`) to be compatible with Keras 3 !
    - The `train` method has been removed in favor of the `fit` method
    - A custom `LossWithMultipleOutputs` has been created to support custom losses providing additional metrics (e.g., the `YOLOLoss`)
    - A new `CheckpointManager` has been designed to mimic the behavior of the previously used `tf.train.CheckpointManager`
    - The `test` method has been removed, but the `evaluate` method has not been overriden yet
    - The `compile` method has been simplified
    - The model serialization now leverages the `keras.serialize_object`, and uses the `deserialize` version for model restoration
    - The data processing pipeline methods have been updated to `prepare_{input / output}` rather than `encode_{input / output}`, and a new `process_batch_{input / output}` is now supported to specifically process batched data (the `map_before_batch` has therefore been removed)
    - A new `_default_{loss / metrics / optimizer}` static variables have been added to reduce the need of overriding the `compile`
- The `datasets` module has been moved to `utils/datasets`, as well as the `hparams` module, moved to `utils/hparams.py`
- The `__init__.py` files in `custom_train_objects/{losses / metrics / optimizers / callbacks}`, as well as `custom_{architectures / layers}` have been modified to follow the same pattern. They now dynamically import custom losses / metrics / ..., without the need to modify these `__init__` files. 
- The new `BaseModel.fit` is properly working with the 3 backends, while leveraging a fully optimized `tf.data` processing pipeline !
- The `Transformers` blocks have been updated, and a working implementation of `Mistral` has been added ! Examples will come in the near future with the `nlp` repository keras 3 update :smile:
- The `custom_layers/multi_head_attention` has been renamed `residual_multi_head_attention` to not confuse with the  `keras.layers.MultiHeadAttention` layer

### Image features

- New algorithms to combine bounding boxes are proposed. They use a connected-components algorithm to compute boxes to combine much faster and in a more reliable way than before
- The `iou` module has been renamed `metrics` to be more generic

### Text features

- The `CTC decoding` methods now leverage the `keras.ctc_decode` function.

## Update 01/05/2024

Only the [data processing](https://github.com/yui-mhcp/data_processing) repository has been updated to enhance keras-3 support. The other projects are in progress ! :smile:

### Major updates

- The `graph_compile` now supports all backend compilation (`tf.function`, `torch.compile` and `jax.jit`) with atomatic detection of `static_argnames` for `JAX`
- The `executing_eagerly` function can detect XLA execution in all backends (by using global variables + `with` statement)
- The `is_tensorflow_graph` allows to differenciate between `tensorflow`-compiled function and `executing_eagerly` which is backend-agnostic
- The `graph_compile` allows `tf.function` compilation regardless of the backend (useful for `tensorflow`-only operations, like `load_audio1`)
- The `graph_compile` and `execute_eagerly` decorators are now unit-tested 
- The `Locality Aware Non-Max-Suppression (LANMS)` is now working with all backends
- `tensorflow` gpu utilities (such as gpu memory usage / memory limitation) are included in `keras_utils/gpu_utils.py`. Other backends will (µhopefullyµ) be supported in future updates if these features are available ;)
- The `utils/thread_utils` has been renamed to `utils/threading`
- The `utils/image/box_utils` has been renamed to `utils/image/bounding_box`
- The `utils/image/bounding_box/geo_utils` has been replaced by a simplified version in `{...}/polygons.py`
- All the `keras.applications.{model}.process_input` have been re-implemented to support `tf.data` regardless of the actual backend
- `tensorflow` and `torch` pass all the unit-tests, and only 2 tests fail in `jax` (due to `XLA`-incompatibility) ! :smile:
- [Work In Progress] The `tensorflow` library is not loaded by default when using other backends. This feature is still experimental and may not be perfectly working in all scenarios, feel free to open an issue ! The `utils/text` module still requires `tensorflow`
- [Work In Progress] The `datasets` module (previously at the root of the directory) is moving to `utils/datasets`. This solves the `datasets` library from `huggingface` import. The dataset processing functions are moving to individual files to facilitate new dataset integration, and clarity of the code 

### Known issues

- The `utils/text` module still imports `tensorflow`
- The `scores` argument in `nms` is currently not supported at all due to errors in `XLA` mode
- The `lanms` is not properly workin in `jax.jit`
- The `max_slice` argument in `distance` is not supported in `jax.jit`
- The CTC decoding methods are only supported in `tensorflow`. `keras>=3.3` have compatible ctc decoding methods.

## Update 01/04/2024

**Only the [data processing](https://github.com/yui-mhcp/data_processing) repository has been updated with experimental keras-3 support**

### Major updates

- **The data processing module has been completely recoded to support the keras 3 multi-backend framework !**
- The `utils/keras_utils` creates a convenient interface over keras-3 to add useful features for `tf.data` pipeline support in a backend-agnostic context (see [this notebook](https://github.com/yui-mhcp/data_processing/blob/master/example_custom_operations.ipynb) for more details)
- The bounding box manipulation has been optimized and simplified
- New unit testing methods have been added to better test the multi-backend support along with `tf.data` pipelines
- The K-Means algorithm is now `XLA`-enabled

### Known issues

The project is at an early stage of multi-backend support ! These issues will be solved in next updates, along with the associated model code (which will also be re-implemented) :yum:

- Some functions are not properly working with the JAX backend
- These functions leverage a `tensorflow` function and are not fully backend agnostic yet :
    - the CTC decoding methods
    - The image augmentation methods
    - The standard Non-Maximum Suppression (NMS)
- Some image normalization leverage the `keras.applications` method, making them not usable in `tf.data` pipeline with non-tensorflow backend
- The new `graph_compile` (replacing `tf_compile`) currently only supports `tf.function` compilation (other backends are executed eagerly)

## Update 01/02/2024

### Major updates

- The `Transformer` inference methods are now compatible with the [XLA](https://www.tensorflow.org/xla) optimization ! This enables the function to run more than 2 times faster compared to regular graph-mode compilation, and around 10 times faster than eager mode ! The main limitation is that the compilation time is much slower for the 1st call, which is important to take into account
- The custom `tf_compile` decorator progressively replaces the regular `tf.function` decorator in  the code in order to reduce retracing as much as possible, while enabling to easily run thee function with/without XLA or in eager mode (see below for more details)
- The `TextTransformer.{sos / eos / pad}_token` are now `tf.constant` instead of `tf.Variable` to be compatible with `XLA` (see below)
- The `utils/text/bpe.py` has been renamed `utils/text/byte_pair_encoding.py`

## Bugs fixed

- The `color` argument is correctly handled in the `plot` function when passing a `dict` of data
- The `CLIP` and `YOLO` architectures have been modified to remove the `Lambda` layers, which raise exception at restoration time
- The `Whisper` tokenizer is now correctly loaded from the `transformers` library using the `openai/whisper-base` tokenizer
- The `Whisper.filter_logits` function has been moved outside of the class to make it an instance variable, to avoid retracing of the inference method. 

## Some information about XLA and graph mode

These information are based on experimental observations, and comes in addition to the [official tensorflow tutorial](https://www.tensorflow.org/api_docs/python/tf/function) about `tf.function`. I highly recommand you to read the tutorial to better understand the following issues and proposed solutions ;)

1) Re-implementation of `experimental_follow_type_hints`, and the `cast_defaults` feature :

In `tensorflow > 2.10`, the `experimental_follow_type_hints` has been removed, causing multiple retracing when passing regular python objects (`int / float`) as kwargs. 

In the below example, the `show1` function raises 2 retracing in `tensorflow==2.10`, and 5 in higher versions. The `follow_type_hints` enables to mimic the reproduce the `2.10` behavior in newer versions, and causes 2 retracing : 1 for the call with the default argument `fn()`, and 1 for the 1st call with an int `fn(i)`. The 1st retracing is interesting because the default value `2` is not casted, as it is not passed as arg / kwarg to the function ! The `cast_defaults` feature solves this limitation by casting unprovided kwargs with an annotation.

With this feature enabled, the `show2` call only retraces once at the 1st call !

```python
@tf.function(experimental_follow_type_hints = False)
def show1(n : tf.Tensor = 2):
    print('Retracing with {}'.format(n))
    return n ** 2

@tensorflow_utils.tf_compile(follow_type_hints = True, cast_defaults = True)
def show2(n : tf.Tensor = 2):
    print('Retracing with {}'.format(n))
    return n ** 2

for fn in (show1, show2):
    print('{}\nTest with {}\n{}'.format('=' * 50, fn.__name__, '=' * 50))
    fn()
    for i in range(5): fn(i)
```
The `cast_kwargs` argument enables to force casting kwargs that are not`bool / str / callables` (e.g., `int / float`) to enable nested-casting. As an example, the `infer_beam_search` takes as config `num_beams`, which may be casted to `tf.Tensor`. Nonetheless, the `TextTransformer.infer` does not contain this argument, which is therefore not casted via the `follow_type_hints` kwarg, and may cause retracing


2) Differences between the regular graph mode and the `XLA` graph mode

The 1st known issue in tensorflow XLA, is that `int32 tf.Variable` are placed on `CPU`, while the XLA does not support reading from multiple devices (i.e. GPU vs CPU). This is the reasing why token variables in the `TextTransformer` class are now `tf.constant` (placed on `GPU`), and not `tf.Variable`

The 2nd major restriction is that XLA does not support variable-length `Tensor` in a while-loop body, whereas in regular graph mode, it is supported provided the `shape_invariant` kwargs. The workaround, inspired from the `transformers` library, is to pre-compute a full-size tensor of zeros (of size `max_length`), and dynamically update slices using the `dynamic_update_slice` function of tensorflow XLA

The last observation is the *bound methods* that always cause retracing. A *bound method* is an object instance method, which is not handled the same way as *functions* in case of retracing.

As an example, in the case of `Whisper`, passing `self.filter_logits` to the `infer` method, will systematically cause retracing. A contrario, passing the `lambda` function returned by `get_filter` will not cause retracing in the subsequent calls, as the stored lambda function will be identified as *known* (same object) by the retracing procedure

```python
def filter_logits(self, scores, tokens, state, ** _):
    to_remove = self.remove_tokens_with_space if state.state is None else self.remove_tokens
    return timestamp_filter(self, scores, tokens[:, :state.t], to_remove, state)

def get_filter(self):
    return lambda * args, ** kwargs: filter_logits(self, * args, ** kwargs)
```

## New year update 01/01/2024

**Happy new year !**

## Main updates

- The `custom_train_objects/{losses / metrics / optimizers}` have been updated such that custom losses / ... are automatically loaded, avoiding redundant modifications in the `__init__` file ! This is still experimental so please open an issue if it raise an error ;)
- A new `utils.distance.text_distance_method` module have been created to replace the previous `utils.text.f1`. The module comes with a new decorator dedicated to `str`-based distance methods, and manually computes matrix-bassed computation (i.e. compute the distance between all pairs)
- New `unitests` for the `custom_train_objects` module !
- The `utils.thread_utils` has been cleaned to fully remove the `Pipeline` mechanism. A simplified yet more efficiant and safe version of the `Producer` and `Consumer` are still available / used

## Bugs fixed

- The losses and metrics are now correctly deserialized with the official `tensorflow.keras.{metrics / losses}.deserialize` method, solving previous errors
- The `tf.keras.optimizers.legacy.*` are now correctly handled in model restoration (only relevant for `tensorflow > 2.10`)
- Minor bug in `box_functions` solved
- Minor bug in `vlines / hlines` plot solved


## Christmas update 25/12/2023

**Merry Christmas to everyone !** 

### Major updates

- **NEW** The [T5](https://huggingface.co/docs/transformers/model_doc/t5) and the [Falcon](https://huggingface.co/docs/transformers/model_doc/falcon) architectures are now available in the [NLP](https://github.com/yui-mhcp/nlp) project !
- The `BaseSTTModel` has been updated to correctly support the `Whisper` inference. Other models like `Jasper` now use a similar method, which is more suitable and clean than the previous method !
- The `execute_eagerly` is now usable on class methods
- The `TextEncoder` directly implements the `multi_{encode / format / split / ...}` methods, which were originally implemented in the `BaseTextModel`
- The new `copy_methods` decorator has enabled a big cleaning of the `BaseTextModel` by removing all *forwarding methods* (e.g., `encode_text` which is a direct call to `self.text_encoder.encode`)
- A new context `timer` is added, enabling time tracking via a `with` statement, instead of the `{start / stop}_timer` (cf [this notebook](https://github.com/yui-mhcp/data_processing/blob/master/example_generic.ipynb) for an example)
- The custom layers in the `custom_layers` module are not dynamically loaded, meaning that you do not have to modify the `__init__.py` file to export your custom layer ! The same behavior will be implemented in other modules (e.g., `custom_train_objects` and `models`)
- A new `generation_utils` module has been created to handle both `RNN` inference and `Transformers` inference at once with a unified and optimized generation loop ! Some features for the `Transformers` are not supported yet (notably returning the decoder self-attention), as the dimensions change accross steps. This will be solved in future updates, but you can go back to the old generation function if needed (by setting `from custom_architecture.transformers_arch.generation_utils import infer` instead in the `text_transformer_arch.py` file).

### Bugs fixed

- In the `ocr` module, the `stream_video` did not correctly forward kwargs to the `predict` method
- The `execute_eagerly` decorator was not working on class methods
- The `BoundingBox` was not correctly handled by some functions (such as `sort_boxes`)
- The `filter_texts` (applied after each `multi_{...}` function of the `TextEncoder`) now correctly returns an empty result if the `required_idx` does not match the constraints
- The `violinplot` is now correctly supported

### Known issues

Due to recent updates of `tensorflow` (version higher than 2.10), some behaviors may raise exceptions :
- The `model_from_json` does not restore `Lambda` layers (this will not be an issue if the model is a custom class or if it does not contain any `Lambda` layers)
- The `tf.keras.losses.*` have a new `fn` entry in their config, which raise an error when reloading them by calling their constructor (which is the current behavior in the `BaseModel`)

The NLP project is currently refactored, some features are not available anymore, notably the `predict` method.

These issues will be solved in the next (1st january) update :smile:

### Plot features

- The `violinplot` and the `boxplot` now support multi-color boxes / violins
- The `bar` plot supports grouped bars (i.e. multiple bars per feature) ([see here](https://github.com/yui-mhcp/data_processing/blob/master/example_generic.ipynb) for an example)


## Update 01/11/2023

### Major new features

- **NEW** `utils/tensorflow_utils` module that provides many useful features and functions for tensorflow. It notably provides 2 `decorator` :
    - `tf_compile` : an equivalent to `tf.function` with a custom re-implementation of the `experimental_follow_type_hints` parameter, which has been removed from `tensorflow 2.11` and higher
    - `execute_eagerly` : an equivalent to the new `tf.numpy_function` decorator added in `tensorflow 2.14` with shape setting (via `tf.ensure_shape`), which is quite useful after calling `tf.{num}py_function` ;)
- **NEW** `utils/wrapper_utils` module that provides custom `decorator`s for better documentation and customization ! The 2 most powerful ones are :
    - `dispatch_wrapper` : enables to make a function customizable by automatically adding and documenting sub-functions (check `utils/file_utils` or `utils/distance/distance_method` for concrete example use cases)
    - `partial`     : this decorator acts similarly to `functools.partial` with 2 major advantages. 1) it supports class methods, and 2) it copies the function `__doc__`

```python
_loading_fn = {}

@dispatch_wrapper(_loading_fn, 'Filename extension')
def load_data(filename):
    return _lading_fn[filename.split('.')[-1]](filename)

@load_data.dispatch
def load_npy(filename):
    return np.load(filename)

load_data.dispatch(pd.read_csv, 'csv')
load_data.dispatch(partial(pd.read_csv, sep = '\t'), 'tsv')

data = load_data('test_numpy.npy')  # works
data = load_data('test_pandas.csv') # also works
data = load_data('test_pandas.tsv') # also works
data = load_data('test_dict.json')  # will fail with a KeyError
help(load_data) # displays the signature of `load_npy` and `pd.read_csv` with their associated extension inf `_loading_fn` !
```
- **NEW** `unitests` testing scripts fully integrated with the `unittest` standard library ! Simply run `python3 -m unittest discover unitests -v` to ensure correcness of the installation ! :yum:

### Main updates

- Cleaning (+ documentation) of the `utils/image` and `utils/audio` modules
- `tf_read_audio` has been removed as it is useless, thanks to the new `execute_eagerly` decorator (`read_audio` is now transparently callable inside `tf.function`)
- `get_cleaners_fn` has been moved to `utils/text/cleaners` with some code cleaning + new cleaners
- Image normalization schemes have been moved from `models/interfaces/base_image_model` to `utils/image/image_normalization` for code cleaning and simplification
- `utils/stream_utils` now contains functions relative to streaming / iterator, such as the `create_iterator` function
- The `TextEncoder.encode` is now transparently callable inside `tf.function`, which makes `BaseTextModel.tf_encode_text` deprecated, and will be removed in the next update (now simply calls `BaseTextModel.encode_text` ;) 
- The `load_data` now supports image and audio loading (which was not the case due to circular imports) --> the code should also be a bit faster to import !

### Loggers features

- The `loggers` module is now completely independant from the `utils` module !
- The `time_to_string` now displays micro-seconds in case of execution time smaller than 1ms
- The `logging.{start / stop}_timer` is now usable
- The `time_logger` is now directly importable from `loggers` (e.g., `from loggers import time_logger`)
- When adding a new level (e.g., `add_level('DEV', 11)`), the level's name is callable both from `logging` (`logging.dev(msg)`) and from the `Logger` class (`logging.getLogger().dev(msg)`)

### Audio features

- New `play_audio` function which directly calls `sounddevice.play` in a separate thread, which seems more stable than the previous callback-based method
- New `resample_audio` method (originally coded inside `read_audio`)

### Image features

- Fix bug in the `get_resized_shape` when `keep_aspect_ratio = True`
- New experimental `custom_cameras` module that enables streaming phone / tablet screen though the `stream_camera` method

### Text features

- new cleaner for `regex` patterns replacements (`replace_patterns`) in addition to the existing word-based replacement (`replace_words`)

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