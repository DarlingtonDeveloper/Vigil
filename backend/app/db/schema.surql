-- SPEC-02: SurrealDB schema will be defined here

-- SPEC-05: Evaluation table for Opik quality gate results
DEFINE TABLE evaluation SCHEMAFULL;

DEFINE FIELD session_id ON evaluation TYPE string;
DEFINE FIELD score ON evaluation TYPE float;
DEFINE FIELD feedback_type ON evaluation TYPE string;
DEFINE FIELD feedback_detail ON evaluation TYPE string;
DEFINE FIELD evaluator ON evaluation TYPE string;
DEFINE FIELD timestamp ON evaluation TYPE datetime;

DEFINE INDEX idx_evaluation_session ON evaluation FIELDS session_id;
DEFINE INDEX idx_evaluation_evaluator ON evaluation FIELDS evaluator;
