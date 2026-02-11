"""时间衰减计算器模块

为 EntityGraph 节点提供时间感知的置信度衰减功能。
设计原则：使用简单的线性时间分段，不考虑节点类型差异。
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any


class TemporalDecayConfig:
    """时间衰减配置（简化版 - 不分类型）"""

    # 新鲜度评分（线性）
    FRESHNESS_SCORES = {
        timedelta(hours=24): 1.0,  # 24小时内
        timedelta(days=3): 0.8,  # 3天内
        timedelta(days=7): 0.6,  # 1周内
        timedelta(days=30): 0.4,  # 1月内
    }
    DEFAULT_FRESHNESS = 0.2  # 更早的数据

    # 阈值
    MIN_CONFIDENTIAL_LEVEL = 0.1  # 最低置信度
    FRESHNESS_THRESHOLD = 0.4  # 低于此值标记为陈旧


class TemporalDecayCalculator:
    """时间衰减计算器"""

    def __init__(self, config: Optional[TemporalDecayConfig] = None):
        self.config = config or TemporalDecayConfig()

    def calculate_freshness(
        self,
        extracted_at: datetime,
        reference_time: Optional[datetime] = None
    ) -> float:
        """计算数据新鲜度 [0, 1]

        Args:
            extracted_at: 信息提取时间
            reference_time: 参考时间（默认当前时间）

        Returns:
            新鲜度分数 [0, 1]
        """
        if reference_time is None:
            reference_time = datetime.now()

        delta = reference_time - extracted_at

        # 如果提取时间在未来，返回最新
        if delta.total_seconds() < 0:
            return 1.0

        # 按时间范围查找新鲜度分数（从小到大匹配）
        for threshold, score in sorted(
            self.config.FRESHNESS_SCORES.items(),
            key=lambda x: x[0]
        ):
            if delta <= threshold:
                return score

        return self.config.DEFAULT_FRESHNESS

    def update_node_attributes(
        self,
        extracted_at: datetime,
        original_confidential_level: float,
        reference_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        根据时间衰减更新节点属性

        Args:
            extracted_at: 信息提取时间
            original_confidential_level: 原始置信度 [0, 1]
            reference_time: 参考时间（默认当前时间）

        Returns:
            包含更新后属性的字典:
            - temporal_confidence: 衰减后置信度
            - uncertainty: 不确定性
            - status: 状态码 (0/1/2)
            - freshness: 新鲜度分数
        """
        freshness = self.calculate_freshness(extracted_at, reference_time)

        # 核心衰减公式（线性乘法）
        temporal_confidence = max(
            original_confidential_level * freshness,
            self.config.MIN_CONFIDENTIAL_LEVEL
        )

        # uncertainty 是 confidential_level 的反向
        uncertainty = 1.0 - temporal_confidence

        # 根据 temporal_confidence 确定 status
        if temporal_confidence >= 0.7:
            status = 2  # 高置信度
        elif temporal_confidence >= 0.4:
            status = 1  # 低置信度
        else:
            status = 0  # 未知/陈旧

        return {
            "temporal_confidence": temporal_confidence,
            "uncertainty": uncertainty,
            "status": status,
            "freshness": freshness
        }

    def is_stale(
        self,
        extracted_at: datetime,
        reference_time: Optional[datetime] = None
    ) -> bool:
        """
        判断数据是否陈旧

        Args:
            extracted_at: 信息提取时间
            reference_time: 参考时间（默认当前时间）

        Returns:
            True if freshness < FRESHNESS_THRESHOLD (0.4)
        """
        freshness = self.calculate_freshness(extracted_at, reference_time)
        return freshness < self.config.FRESHNESS_THRESHOLD
