from .realenv import RealEnv
from .mppi import MPPIController

all_envmodels = {
	"real":RealEnv
}

def get_envmodel(config):
	return RealEnv
