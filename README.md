# StopWordCriteria

A [Huggingface transformer's](https://huggingface.co/docs/transformers/en/index) [stopping criteria](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.StoppingCriteria) that halts the generation when a given stop word is encountered.

## Installation

Put the `stop_word_criteria.py` file in your project.

## Usage

Here is an example usage, on a single element batch:

```python
# extracting a question
prompt='If I understand you clearly, your question is: "'
stop_words = ['"']

# create the stopping criteria
stopping_criteria = StopWordCriteria(tokenizer=tokenizer, prompts=[prompt], stop_words=stop_words)

# tokenize the prompt
inputs_tokens = self.tokenizer.encode(prompt, return_tensors="pt")

# runs the LLM, producing tokens that represents `inputs_tokens + generated_text + stopword + maybe more`
output_tokens = self.model.generate(inputs_tokens, stopping_criteria=[stopping_criteria])

# extract the generated text from output tokens, cutting the prompt and stop words
question = stopping_criteria.extract_answers(output_tokens, strip_stopword=True)[0]
```

## Implementation details

The same stop word can be mapped to various tokens depending on context, we thus have to decode it into a string before checking for the presence of a stop word.

As running the stopping criteria whenever a token is created could be costly[^cost], the class constructor has a `check_every` parameter that defaults to 1.
Setting it to 1 means that we run the stopping criteria every iterations and will stop as soon as a stop word is generated.
Setting it to a high value will run the stopping criteria periodically, meaning that we might generate several tokens after the stop word.
Either way, the `extract_answers` function will remove all text generated after the first stop word encountered.

[^cost]: This depends on how much time it takes your model to generate a token.

## Credits

This implementation was inspired by a [Huggingface discussion thread](https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/9) as well as [Outlines's implementation of a similar stopping concept](https://github.com/outlines-dev/outlines/blob/main/outlines/generate/api.py).
