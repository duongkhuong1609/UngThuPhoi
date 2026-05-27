export type FormState = {
  image: File | null;
  imagePreview: string | null;
  sample_path: string | null;
  localization_sample_path: string | null;
  patient_name: string;
  age: string;
  sex: '' | 'female' | 'male';
  smoking_status: '' | 'never' | 'former' | 'current';
  tumor_size: string;
  family_history: '' | 'no' | 'yes';
  symptom_score: string;
};

export type PredictionResponse = {
  risk_level: 'high_malignancy' | 'medium_risk' | 'low_malignancy';
  predicted_class_index: number;
  predicted_label?: 'high_malignancy' | 'medium_risk' | 'low_malignancy';
  calibrated_probability: number;
  confidence: number;
  probabilities: {
    low_malignancy: number;
    medium_risk: number;
    high_malignancy: number;
  };
  conclusion: string;
  warning: string | null;
  warnings?: string[];
  temperature: number;
  checkpoint: string;
  history_id?: string | null;
  debug?: {
    sample_path_requested?: string | null;
    sample_path_resolved?: string | null;
    upload_filename?: string | null;
    image_sha1_16?: string | null;
    raw_logits?: number[];
    softmax_probabilities?: Record<string, number>;
    predicted_class_index?: number;
    predicted_label?: string;
    display_label?: string;
    class_names_by_index?: string[];
  };
  localization?: {
    available: boolean;
    error?: string | null;
    source?: string;
    checkpoint?: string;
    conf_threshold?: number;
    image_size?: number;
    box_count: number;
    boxes: Array<{
      id: number;
      class_id: number;
      label: string;
      confidence: number;
      xyxy: number[];
    }>;
    annotated_image?: string | null;
  };
  input_summary: {
    patient_name?: string;
    patient_id?: string;
    age: string;
    sex: string;
    smoking_status: string;
    tumor_size: string;
    tumor_size_effective?: string | null;
    tumor_size_imputed?: boolean;
    tumor_size_missing?: boolean;
    tumor_size_missing_encoded?: boolean;
    family_history: string;
    symptom_score: string;
    localization_sample_path?: string | null;
  };
  tabular_preprocessing?: {
    features?: Record<string, number>;
    imputed?: Record<string, number>;
    missing_encoded?: Record<string, {
      raw_reference_value?: number;
      scaled_value?: number;
      strategy?: string;
    }>;
    missing_fields?: string[];
    invalid_fields?: string[];
    imputation_strategy?: {
      type?: string;
      reason?: string;
    };
  };
  localization_classification_consistency?: {
    no_box_high_risk: boolean;
    message?: string | null;
  };
};

export type ValidationErrors = Partial<Record<keyof Omit<FormState, 'imagePreview' | 'sample_path' | 'localization_sample_path'>, string>> & {
  image?: string;
};

export type PredictionHistoryRecord = {
  id: string;
  patient_name: string;
  patient_id: string;
  age: string;
  sex: string;
  smoking_status: string;
  tumor_size: string;
  tumor_size_effective?: string | number | null;
  tumor_size_imputed?: boolean;
  tumor_size_missing?: boolean;
  tumor_size_missing_encoded?: boolean;
  family_history: string;
  symptom_score: string;
  image_path: string;
  source: 'sample' | 'upload' | string;
  predicted_label: 'high_malignancy' | 'medium_risk' | 'low_malignancy' | string;
  predicted_class_index: number;
  probabilities: {
    low_malignancy: number;
    medium_risk: number;
    high_malignancy: number;
  };
  confidence: number;
  temperature: number;
  checkpoint: string;
  timestamp: string;
  sample_path?: string | null;
  upload_filename?: string | null;
  localization?: {
    source?: string | null;
    sample_path?: string | null;
    result?: {
      available?: boolean;
      error?: string | null;
      box_count?: number;
      annotated_image?: string | null;
    };
  };
};
