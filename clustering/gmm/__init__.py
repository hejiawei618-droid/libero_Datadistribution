"""
gmm 模块: GMM 峰数判断 (BIC 准则) + 统计汇总

    Step 7: gmm_select_peaks()   → all_best_k, bic_curves
    Step 8: final_statistics()    → counter, summary
"""

from .step7_gmm import gmm_select_peaks, summarize_peaks, save_gmm_results, load_gmm_results
from .step8_statistics import final_statistics, save_statistics, load_statistics
