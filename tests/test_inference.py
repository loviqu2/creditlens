import sys
sys.path.append('.')
from src.inference import get_risk_tier


#test out the risk tier using git actions 
def test_risk_tier_boundaries():
    assert get_risk_tier(0.10) == "Low Risk"
    assert get_risk_tier(0.20) == "Moderate Risk"
    assert get_risk_tier(0.40) == "High Risk"
    assert get_risk_tier(0.70) == "Very High Risk"