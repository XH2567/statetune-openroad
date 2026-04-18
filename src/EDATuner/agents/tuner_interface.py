# tuner_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class TunerAgent(ABC):
    """
    Base class for tuner agents.
    Subclasses may be Bayesian optimizers or LLM-based agents (for example GPT-4 or Llama 3).
    """
    
    @abstractmethod
    def suggest_params(self, history: List[Dict]) -> Dict[str, Any]:
        """
        Suggest the next parameter set based on historical data.
        Args:
            history: list of previous runs containing parameters and results
                     [{'params': {...}, 'metrics': {...}}, ...]
        Returns:
            next_params: parameter dictionary for the next trial
        """
        pass

    @abstractmethod
    def update(self, params: Dict, metrics: Dict):
        """
        Provide the latest trial result to the agent so it can update its internal model (for example a GP model or LLM context).
        """
        pass
