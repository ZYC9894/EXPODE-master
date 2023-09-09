from .q_learner import QLearner
from .qatten_learner import QattenLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .qtran_transformation_learner import QLearner as QTranTRANSFORMATIONLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner

from .fast_q_learner import fast_QLearner

from .qplex_curiosity_vdn_learner import QPLEX_curiosity_vdn_Learner

from .EXPODE_curiosity_learner import EXPODE_curiosity_Learner



REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qatten_learner"] = QattenLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["qtran_transformation_learner"] = QTranTRANSFORMATIONLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner


REGISTRY["fast_QLearner"] = fast_QLearner
REGISTRY['qplex_curiosity_vdn_learner']=QPLEX_curiosity_vdn_Learner

REGISTRY['EXPODE_curiosity_learner']=EXPODE_curiosity_Learner



