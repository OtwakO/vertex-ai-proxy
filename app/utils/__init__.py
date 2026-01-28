# Utils package initialization

from app.utils.api_key import APIKeyManager, test_api_key
from app.utils.cache import (
    ResponseCacheManager,
    generate_cache_key,
    generate_cache_key_all,
)
from app.utils.error_handling import (
    handle_api_error,
    handle_gemini_error,
    translate_error,
)
from app.utils.logging import format_log_message, log, log_manager, logger
from app.utils.version import check_version
from app.utils.maintenance import handle_exception, schedule_cache_cleanup
from app.utils.rate_limiting import protect_from_abuse
from app.utils.request import ActiveRequestsManager
from app.utils.response import openAI_from_text
from app.utils.stats import clean_expired_stats, update_api_call_stats
