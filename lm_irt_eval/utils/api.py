from openai_harmony import load_harmony_encoding, HarmonyEncodingName


OPENAI_ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


# Custom Exception for non-retriable HTTP errors
class NonRetriableHTTPError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code
