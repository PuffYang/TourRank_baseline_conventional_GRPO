from verl.tools.dr_tulu_search_tools import GoogleSearchTool


def test_extract_result_records_prefers_search_like_records():
    payload = {
        "data": [
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "content": "raw"},
        ],
        "organic": [
            {
                "title": "Latest AI News",
                "link": "https://example.com/ai-news",
                "snippet": "Major model launch and policy updates.",
            }
        ],
    }

    records = GoogleSearchTool._extract_result_records(payload)

    assert records == payload["organic"]


def test_extract_result_records_ignores_unrelated_dict_lists():
    payload = {
        "data": [
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "content": "raw"},
        ]
    }

    records = GoogleSearchTool._extract_result_records(payload)

    assert records == []
