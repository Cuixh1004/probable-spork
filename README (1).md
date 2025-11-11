# Food Delivery Coupon Strategy Pipeline

## Overview

This is a comprehensive end-to-end automated pipeline for analyzing and simulating food delivery platform coupon expansion and traffic distribution algorithms. It implements three core components:

1. **Coupon Expansion Algorithm** - Dynamic interval random algorithm with trigger conditions
2. **Traffic Distribution Model** - Magic Cube 5-dimensional context-aware ranking system
3. **User Segmentation Engine** - RFM (Recency, Frequency, Monetary) analysis and clustering

## Architecture

```
Pipeline Components:
├── coupon_expansion.py      # Coupon expansion algorithm (膨胀红包逻辑)
├── traffic_distribution.py  # Magic Cube traffic distribution model (流量分发算法)
├── user_segmentation.py     # RFM user segmentation (用户分层)
└── pipeline.py              # End-to-end orchestration (管道编排)
```

## Key Features

### 1. Coupon Expansion Algorithm

Implements dynamic coupon expansion based on:
- **User Type**: New users (80% probability), Active users (50%), Inactive users (20%)
- **Trigger Conditions**:
  - Consecutive orders (3+ in 7 days)
  - High consumption threshold (≥100 RMB)
  - Peak activity hours (11-13, 17-19)
  - Cross-category consumption
  - Social sharing

**Output**: Expansion multiple (1.0x - 3.0x) for each user

### 2. Magic Cube Traffic Distribution Model

5-dimensional context-aware ranking system:
- **User Dimension (35%)**: RFM score, historical CTR, preference match
- **Time Dimension (20%)**: Peak hours, weekday/weekend
- **Location Dimension (15%)**: City, district, distance
- **Entrance Dimension (15%)**: Homepage, search, category, recommendation, push, banner
- **Business Dimension (15%)**: Category match, merchant quality

**Output**: Relevance score, CTR/CVR prediction, ROI calculation

### 3. User Segmentation (RFM Model)

8 user segments based on Recency, Frequency, Monetary:
- **Champions**: High R, F, M (最有价值客户)
- **Loyal Customers**: Low R, High F, M (忠诚客户)
- **Potential Loyalists**: Recent, Medium F, M (潜在忠诚客户)
- **At Risk**: High R, High F, M (风险客户)
- **Can't Lose Them**: High R, High F, M (不能失去的客户)
- **New Customers**: Recent, Low F, M (新客户)
- **Need Attention**: Medium R, F, M (需要关注)
- **About to Sleep**: High R, Low F, M (即将流失)

**Output**: Segment classification + recommended actions

## Usage

### Basic Pipeline Execution

```python
from pipeline import EndToEndPipeline

# Initialize pipeline
pipeline = EndToEndPipeline(
    base_coupon_value=10.0,  # 10 RMB base coupon
    num_users=1000,          # Process 1000 users
    num_merchants=100,       # 100 merchants
)

# Run full pipeline
results = pipeline.run_full_pipeline()

# Save results
pipeline.save_results('/path/to/output.json')
```

### Individual Component Usage

#### Coupon Expansion

```python
from models.coupon_expansion import CouponExpansionAlgorithm, CouponExpansionConfig

config = CouponExpansionConfig(base_coupon_value=10.0)
algorithm = CouponExpansionAlgorithm(config)

# Process single user
expansion_multiple = algorithm.calculate_expansion_multiple(user, triggers)
final_coupon = algorithm.get_final_coupon_value(user, triggers)

# Batch process
results = algorithm.batch_process_users(users, timestamp, user_orders)
```

#### Traffic Distribution

```python
from models.traffic_distribution import MagicCubeModel

model = MagicCubeModel(base_ctr=0.05, base_cvr=0.10, coupon_value=10.0)

# Single allocation
allocation = model.allocate_traffic(
    coupon_id='coupon_001',
    user_profile=user_profile,
    merchant_profile=merchant_profile,
    timestamp=timestamp,
    entrance_type='homepage',
)

# Batch allocation
allocations = model.batch_allocate(
    coupon_id='coupon_001',
    user_profiles=user_profiles,
    merchant_profile=merchant_profile,
    timestamp=timestamp,
    entrance_type='homepage',
)
```

#### User Segmentation

```python
from models.user_segmentation import RFMModel

model = RFMModel()

# Segment single user
rfm_score = model.calculate_rfm_score(
    days_since_last_order=10,
    num_orders=5,
    total_spending=500,
)

# Batch segment
results = model.batch_segment_users(user_data)

# Get segment info
segment_info = model.get_segment_characteristics(segment)
```

## Output Structure

Pipeline outputs a comprehensive JSON file with:

```json
{
  "expansion_results": [
    {
      "user_id": "user_001",
      "base_coupon": 10.0,
      "final_coupon": 25.0,
      "expansion_multiple": 2.5,
      "triggers": ["consecutive_orders", "peak_activity"],
      "is_expanded": true,
      "timestamp": 1700000000
    }
  ],
  "traffic_allocations": [
    {
      "coupon_id": "coupon_001",
      "user_id": "user_001",
      "relevance_score": 0.75,
      "is_allocated": true,
      "expected_ctr": 0.065,
      "expected_cvr": 0.124,
      "expected_roi": 0.24
    }
  ],
  "user_segments": [
    {
      "user_id": "user_001",
      "segment": "champions",
      "rfm_score": 85.5,
      "recommended_actions": [...]
    }
  ],
  "statistics": {
    "expansion": {
      "expansion_rate": 0.497,
      "avg_expansion_multiple": 1.61,
      "total_users": 1000
    },
    "traffic": {
      "allocation_rate": 0.9988,
      "avg_ctr": 0.0653,
      "avg_cvr": 0.1246,
      "avg_roi": 0.25
    },
    "segmentation": {
      "champions": {"count": 50, "percentage": 5.0},
      "loyal_customers": {"count": 120, "percentage": 12.0}
    }
  },
  "recommendations": {
    "expansion_optimization": [...],
    "traffic_optimization": [...],
    "user_targeting": [...]
  }
}
```

## Key Metrics

### Coupon Expansion Metrics
- **Expansion Rate**: % of users receiving expanded coupons
- **Avg Expansion Multiple**: Average coupon value multiplier
- **Trigger Distribution**: Which conditions drive expansion

### Traffic Distribution Metrics
- **Allocation Rate**: % of user-merchant pairs allocated traffic
- **Avg CTR**: Expected click-through rate
- **Avg CVR**: Expected conversion rate
- **Avg ROI**: Return on investment per coupon

### User Segmentation Metrics
- **Segment Distribution**: % of users in each segment
- **Avg RFM Score**: Overall user value score
- **Segment-specific Metrics**: Characteristics per segment

## Recommendations Engine

The pipeline automatically generates actionable recommendations:

1. **Expansion Optimization**
   - Adjust expansion probabilities based on current rates
   - Optimize expansion multiple ranges
   - Fine-tune trigger conditions

2. **Traffic Optimization**
   - Adjust allocation thresholds
   - Optimize dimension weights
   - Improve context modeling

3. **User Targeting**
   - Segment-specific strategies
   - Retention programs for at-risk users
   - Activation programs for new users

## Performance Considerations

- **Scalability**: Designed for 1000+ users, 100+ merchants
- **Processing Time**: ~30-60 seconds for 1000 users
- **Memory**: ~500MB for standard configuration
- **Parallelization**: Can be extended with multiprocessing for larger datasets

## Configuration Parameters

### Coupon Expansion Config
```python
CouponExpansionConfig(
    base_coupon_value=10.0,           # Base coupon amount (RMB)
    min_expansion_multiple=1.0,       # Minimum expansion multiplier
    max_expansion_multiple=3.0,       # Maximum expansion multiplier
    new_user_expansion_probability=0.8,      # New user expansion rate
    active_user_expansion_probability=0.5,   # Active user expansion rate
    inactive_user_expansion_probability=0.2, # Inactive user expansion rate
)
```

### Magic Cube Model Config
```python
MagicCubeModel(
    base_ctr=0.05,        # Base click-through rate
    base_cvr=0.10,        # Base conversion rate
    coupon_value=10.0,    # Coupon value (RMB)
)
```

### RFM Model Config
```python
RFMModel(
    recency_weight=0.3,    # Weight for recency dimension
    frequency_weight=0.35, # Weight for frequency dimension
    monetary_weight=0.35,  # Weight for monetary dimension
)
```

## Future Enhancements

1. **Real Data Integration**: Connect to actual platform APIs
2. **Advanced ML Models**: Deep learning for CTR/CVR prediction
3. **A/B Testing Framework**: Automated experiment design and analysis
4. **Real-time Streaming**: Process events in real-time
5. **Multi-objective Optimization**: Balance multiple KPIs
6. **Causal Inference**: Measure true impact of interventions

## References

- Meituan: Scenario-Aware Ranking Model for Personalized Recommender System
- Alibaba: Ele.me Red Packet Strategy and Traffic Distribution
- PKU Guanghua: Food Delivery Coupon Impact Analysis
- RFM Analysis: Customer Segmentation and Targeting

## License

Internal Use Only
