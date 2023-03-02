from dataclasses import dataclass


@dataclass(kw_only=True)
class WebcamScraperConfig:
    api_key: str
    webcam_count: int
    dst_dir: str

    async_download: bool = True
    webcam_timeframe: str = "year"

    use_record_keeper_page_params: bool = False

    request_limit: int = 50
    request_offset: int = 0

    download_max_concurrent_requests: int = 5
    download_request_timeout: int = 60
