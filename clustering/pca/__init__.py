"""
pca 模块: LIBERO_OBJECT 数据提取与 PCA 降维

    Step 1: extract_states_and_actions()  → states, actions
    Step 2: reduce_states_pca()           → states_pca, pca
"""

from .step1_extract import extract_states_and_actions
from .step2_pca import reduce_states_pca, save_pca_results
