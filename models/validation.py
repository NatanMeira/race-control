from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass


class CompoundEnum(Enum):
    """
    Enum para compostos de pneus válidos na F1.
    """
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"
    
    @classmethod
    def get_valid_values(cls) -> List[str]:
        """Retorna lista de valores válidos."""
        return [compound.value for compound in cls]
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Verifica se o valor é um composto válido."""
        return value in cls.get_valid_values()


class ConfidenceEnum(Enum):
    """
    Enum para níveis de confiança da predição.
    """
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


@dataclass
class ValidationResult:
    """
    Resultado de validação de dados.
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class PredictionRequestValidator:
    """
    Validador para requisições de predição.
    """
    
    REQUIRED_FIELDS = ['TyreLife', 'LapTime_Delta', 'Cumulative_Degradation', 'Position', 'Compound']
    
    # Limites razoáveis baseados na F1
    TYRE_LIFE_MAX = 60  # Máximo de voltas
    LAP_TIME_DELTA_MAX = 10.0  # Máximo delta em segundos
    DEGRADATION_MAX = 50.0  # Máximo índice de degradação
    POSITION_MIN = 1
    POSITION_MAX = 20  # Máximo pilotos na grid
    
    @classmethod
    def validate(cls, data: Dict[str, Any]) -> ValidationResult:
        """
        Valida os dados de entrada para predição.
        """
        errors = []
        warnings = []
        
        # Verificar campos obrigatórios
        missing_fields = [field for field in cls.REQUIRED_FIELDS if field not in data]
        if missing_fields:
            errors.append(f"Campos obrigatórios ausentes: {missing_fields}")
            return ValidationResult(is_valid=False, errors=errors)
        
        try:
            # Validar TyreLife
            tyre_life = int(data['TyreLife'])
            if tyre_life < 0:
                errors.append("TyreLife deve ser um valor positivo")
            elif tyre_life > cls.TYRE_LIFE_MAX:
                warnings.append(f"TyreLife ({tyre_life}) é excepcionalmente alto (>{cls.TYRE_LIFE_MAX})")
            
            # Validar LapTime_Delta
            lap_time_delta = float(data['LapTime_Delta'])
            if lap_time_delta > cls.LAP_TIME_DELTA_MAX:
                warnings.append(f"LapTime_Delta ({lap_time_delta}) é muito alto (>{cls.LAP_TIME_DELTA_MAX})")
            
            # Validar Cumulative_Degradation
            degradation = float(data['Cumulative_Degradation'])
            if degradation < 0:
                errors.append("Cumulative_Degradation deve ser um valor não-negativo")
            elif degradation > cls.DEGRADATION_MAX:
                warnings.append(f"Cumulative_Degradation ({degradation}) é excepcionalmente alta (>{cls.DEGRADATION_MAX})")
            
            # Validar Position
            position = int(data['Position'])
            if position < cls.POSITION_MIN or position > cls.POSITION_MAX:
                errors.append(f"Position deve estar entre {cls.POSITION_MIN} e {cls.POSITION_MAX}")
            
            # Validar Compound
            compound = str(data['Compound']).upper()
            if not CompoundEnum.is_valid(compound):
                errors.append(f"Compound deve ser um dos valores: {CompoundEnum.get_valid_values()}")
                
        except (ValueError, TypeError) as e:
            errors.append(f"Erro de tipo de dados: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )