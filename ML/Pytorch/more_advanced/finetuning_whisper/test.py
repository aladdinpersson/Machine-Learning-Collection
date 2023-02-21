from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained(
    f"openai/whisper-tiny", task="transcribe"
)
encoded_string = tokenizer.encode("")[0]
print(encoded_string) # should print 50258
print(tokenizer.bos_token_id) # should print 50257
