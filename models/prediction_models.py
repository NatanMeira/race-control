from dataclasses import dataclass
from typing import Optional, Any, Dict
from enum import Enum
import pandas as pd


@dataclass
class PredictionRequest:
    """
    Modelo de entrada para predição de pit stop.
    Representa os dados de telemetria necessários para análise.
    """
    TyreLife: int
    LapTime_Delta: float
    Cumulative_Degradation: float
    Position: int
    Compound: str
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Converte o modelo para DataFrame na ordem correta das features.
        """
        data = {
            'TyreLife': [self.TyreLife],
            'LapTime_Delta': [self.LapTime_Delta],
            'Cumulative_Degradation': [self.Cumulative_Degradation],
            'Position': [self.Position],
            'Compound': [self.Compound]
        }
        return pd.DataFrame(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionRequest':
        """
        Cria uma instância a partir de um dicionário.
        """
        return cls(
            TyreLife=int(data['TyreLife']),
            LapTime_Delta=float(data['LapTime_Delta']),
            Cumulative_Degradation=float(data['Cumulative_Degradation']),
            Position=int(data['Position']),
            Compound=str(data['Compound'])
        )


@dataclass
class PredictionResponse:
    """
    Modelo de resposta da predição de pit stop.
    """
    prediction: int
    message: str
    confidence: str
    telemetry_summary: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte o modelo para dicionário para serialização JSON.
        """
        result = {
            'prediction': self.prediction,
            'message': self.message,
            'confidence': self.confidence
        }
        if self.telemetry_summary:
            result['telemetry_summary'] = self.telemetry_summary
        return result


@dataclass
class ApiResponse:
    """
    Modelo genérico para respostas da API.
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário para serialização JSON.
        """
        result = {'success': self.success}
        if self.data is not None:
            result['data'] = self.data
        if self.error:
            result['error'] = self.error
        if self.message:
            result['message'] = self.message
        return result


class StatusEnum(Enum):
    """
    Enum para status da aplicação.
    """
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    LOADING = "loading"