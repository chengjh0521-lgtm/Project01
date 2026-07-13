import json
import logging
import os

import requests


def openai_call(
        apikey,
        model="gpt-3.5-turbo",
        user_content="How do I make tomato beef stew?",
        system_content=None):
    base_url = None
    if model.startswith("deepseek"):
        base_url = "https://api.deepseek.com"
        apikey = apikey or os.environ.get("DEEPSEEK_API_KEY")
    elif model.startswith("gpt-3.5-turbo"):
        base_url = "https://api.moonshot.cn/v1"
    else:
        apikey = apikey or os.environ.get("OPENAI_API_KEY")

    if not apikey:
        return "API key is required. Paste your DeepSeek API key or set DEEPSEEK_API_KEY."

    apikey = str(apikey).strip()
    try:
        apikey.encode("ascii")
    except UnicodeEncodeError:
        return (
            "LLM inference failed: API key contains non-ASCII characters. "
            "Please paste only the raw DeepSeek key, without Chinese labels, quotes, or spaces."
        )

    if system_content is not None and len(system_content.strip()):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [{"role": "user", "content": user_content}]

    try:
        # Keep request headers ASCII-only. Chinese subtitle prompts are sent in
        # an ASCII JSON body (with Unicode escapes), which OpenAI-compatible
        # services decode as UTF-8 and avoids client/proxy ASCII failures.
        endpoint = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        payload = json.dumps(
            {"model": model, "messages": messages},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        response = requests.post(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {apikey}",
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
            timeout=(15, 120),
        )
        response.raise_for_status()
        response_data = response.json()
        result = response_data["choices"][0]["message"]["content"]
        logging.info("OpenAI-compatible model inference done.")
        return result
    except Exception as exc:
        logging.exception("OpenAI-compatible model inference failed.")
        return f"LLM inference failed: {exc}"


if __name__ == "__main__":
    from llm.demo_prompt import demo_prompt

    print(openai_call(os.environ.get("OPENAI_API_KEY"), "gpt-3.5-turbo-0125", demo_prompt))
