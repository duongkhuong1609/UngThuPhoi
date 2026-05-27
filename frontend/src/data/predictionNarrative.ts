type RiskLevel = 'high_malignancy' | 'medium_risk' | 'low_malignancy' | string;

type ProbabilityMap = {
  low_malignancy: number;
  medium_risk: number;
  high_malignancy: number;
};

type PatientNarrativeInput = {
  riskLevel: RiskLevel;
  probabilities: ProbabilityMap;
  age?: string | number;
  smokingStatus?: string;
  familyHistory?: string;
  symptomScore?: string | number;
  tumorSize?: string | number | null;
  tumorSizeMissing?: boolean;
  tumorSizeImputed?: boolean;
  localizationAvailable?: boolean;
  localizationBoxCount?: number | null;
  noBoxHighRisk?: boolean;
};

type Leaning = 'high' | 'low' | 'unclear' | null;

export type PredictionNarrative = {
  heading: string;
  explanation: string;
  supportingNote?: string | null;
  advice: string[];
  leaning: Leaning;
};

export function labelRisk(value: RiskLevel): string {
  if (value === 'high_malignancy') return 'Nguy cơ cao';
  if (value === 'medium_risk') return 'Nguy cơ trung gian';
  if (value === 'low_malignancy') return 'Nguy cơ thấp';
  return value;
}

function parseNumber(value: string | number | null | undefined): number | null {
  if (value === null || value === undefined || value === '') return null;
  const parsed = typeof value === 'number' ? value : Number(String(value).replace(',', '.'));
  return Number.isFinite(parsed) ? parsed : null;
}

function factorSummary(input: PatientNarrativeInput): string[] {
  const factors: string[] = [];
  const age = parseNumber(input.age);
  const symptomScore = parseNumber(input.symptomScore);
  const tumorSize = parseNumber(input.tumorSize);

  if (age !== null && age >= 60) factors.push('tuổi từ 60 trở lên');
  if (input.smokingStatus === 'current') factors.push('đang hút thuốc');
  else if (input.smokingStatus === 'former') factors.push('đã từng hút thuốc');
  if (input.familyHistory === 'yes') factors.push('có tiền sử gia đình');
  if (symptomScore !== null && symptomScore >= 6) factors.push('điểm triệu chứng đang ở mức khá cao');
  if (tumorSize !== null && tumorSize >= 20) factors.push('kích thước nốt ác tính khá lớn');

  return factors;
}

function naturalJoin(items: string[]): string {
  if (items.length === 0) return '';
  if (items.length === 1) return items[0];
  if (items.length === 2) return `${items[0]} và ${items[1]}`;
  return `${items.slice(0, -1).join(', ')} và ${items[items.length - 1]}`;
}

function determineLeaning(input: PatientNarrativeInput): Leaning {
  if (input.riskLevel !== 'medium_risk') return null;
  const { high_malignancy, low_malignancy } = input.probabilities;
  const gap = Math.abs(high_malignancy - low_malignancy);
  if (gap < 0.08) return 'unclear';
  return high_malignancy > low_malignancy ? 'high' : 'low';
}

function buildHealthAdvice(input: PatientNarrativeInput, leaning: Leaning, factors: string[]): string[] {
  const advice: string[] = [];
  const localized = Boolean(input.localizationAvailable);
  const localizedCount = Number(input.localizationBoxCount ?? 0);
  const symptomScore = parseNumber(input.symptomScore);
  const smokingStatus = input.smokingStatus;
  const tumorSizeMissing = Boolean(input.tumorSizeMissing);
  const tumorSizeImputed = Boolean(input.tumorSizeImputed);

  if (input.riskLevel === 'high_malignancy') {
    advice.push('Nên sắp xếp thăm khám chuyên khoa sớm và mang theo ảnh CT cùng kết quả này để bác sĩ đối chiếu.');
  } else if (input.riskLevel === 'medium_risk') {
    advice.push(
      leaning === 'high'
        ? 'Nên theo dõi sát hơn và ưu tiên tái khám sớm nếu có thêm triệu chứng hoặc cảm thấy sức khỏe thay đổi.'
        : leaning === 'low'
          ? 'Nên tiếp tục theo dõi theo hẹn và không nên chủ quan nếu xuất hiện dấu hiệu bất thường mới.'
          : 'Nên giữ lịch theo dõi đều và trao đổi thêm với bác sĩ nếu muốn làm rõ kết quả.'
    );
  } else {
    advice.push('Nên duy trì theo dõi sức khỏe định kỳ và vẫn giữ các lần tái khám theo hẹn của bác sĩ.');
  }

  if (smokingStatus === 'current') {
    advice.push('Nên giảm và tiến tới ngừng hút thuốc càng sớm càng tốt để giảm thêm nguy cơ cho phổi.');
  } else if (smokingStatus === 'former') {
    advice.push('Nên tiếp tục duy trì việc không hút thuốc và tránh môi trường có nhiều khói thuốc.');
  }

  if (symptomScore !== null && symptomScore >= 6) {
    advice.push('Nếu ho kéo dài, đau ngực, khó thở hoặc mệt nhiều hơn, nên đi khám sớm thay vì chờ đến lịch hẹn kế tiếp.');
  }

  if (localized && localizedCount > 0) {
    advice.push(`Ảnh hiện có ${localizedCount} vùng nghi ngờ được khoanh, nên lưu lại ảnh này để bác sĩ xem đúng vị trí cần chú ý.`);
  }

  if (tumorSizeMissing || tumorSizeImputed) {
    advice.push('Nếu có thể, nên bổ sung thêm kích thước nốt ác tính để hệ thống ước lượng sát hơn ở lần dự đoán sau.');
  }

  if (factors.length > 0) {
    advice.push(`Những yếu tố đang cần lưu ý thêm gồm ${naturalJoin(factors)}.`);
  }

  return advice;
}

export function buildPredictionNarrative(input: PatientNarrativeInput): PredictionNarrative {
  const leaning = determineLeaning(input);
  const factors = factorSummary(input);
  const tumorSizeMissing = Boolean(input.tumorSizeMissing);
  const tumorSizeImputed = Boolean(input.tumorSizeImputed);
  const advice = buildHealthAdvice(input, leaning, factors);

  if (input.riskLevel === 'high_malignancy') {
    return {
      heading: 'Nguy cơ ung thư cao',
      explanation:
        tumorSizeMissing || tumorSizeImputed
          ? 'Kết quả hiện nghiêng rõ về nguy cơ cao. Có thể bổ sung thêm kích thước nốt ác tính để lần đánh giá sau sát hơn.'
          : 'Kết quả hiện nghiêng rõ về nguy cơ cao.',
      supportingNote: input.noBoxHighRisk
        ? 'Ảnh khoanh vùng chưa thật sự rõ, nên cần đối chiếu thêm với bác sĩ khi đọc kết quả.'
        : null,
      advice,
      leaning,
    };
  }

  if (input.riskLevel === 'low_malignancy') {
    return {
      heading: 'Nguy cơ ung thư thấp',
      explanation:
        tumorSizeMissing || tumorSizeImputed
          ? 'Kết quả hiện nghiêng về nguy cơ thấp. Có thể bổ sung thêm kích thước nốt ác tính để lần đánh giá sau đầy đủ hơn.'
          : 'Kết quả hiện nghiêng về nguy cơ thấp.',
      supportingNote: null,
      advice,
      leaning,
    };
  }

  if (leaning === 'high') {
    return {
      heading: 'Nguy cơ ung thư trung gian, xu hướng nghiêng về nguy cơ cao',
      explanation:
        tumorSizeMissing || tumorSizeImputed
          ? 'Kết quả vẫn thuộc nhóm trung gian nhưng đang nhỉnh về phía nguy cơ cao. Có thể bổ sung thêm kích thước nốt ác tính để hệ thống ước lượng sát hơn.'
          : 'Kết quả vẫn thuộc nhóm trung gian nhưng đang nhỉnh về phía nguy cơ cao.',
      supportingNote: null,
      advice,
      leaning,
    };
  }

  if (leaning === 'low') {
    return {
      heading: 'Nguy cơ ung thư trung gian, xu hướng nghiêng về nguy cơ thấp',
      explanation:
        tumorSizeMissing || tumorSizeImputed
          ? 'Kết quả vẫn thuộc nhóm trung gian nhưng đang nhỉnh về phía nguy cơ thấp. Có thể bổ sung thêm kích thước nốt ác tính để hệ thống ước lượng sát hơn.'
          : 'Kết quả vẫn thuộc nhóm trung gian nhưng đang nhỉnh về phía nguy cơ thấp.',
      supportingNote: null,
      advice,
      leaning,
    };
  }

  return {
    heading: 'Nguy cơ ung thư trung gian, chưa nghiêng rõ về nguy cơ thấp hay cao',
    explanation:
      tumorSizeMissing || tumorSizeImputed
        ? 'Kết quả đang ở vùng trung gian và chênh lệch giữa hai phía còn khá gần nhau. Có thể bổ sung thêm kích thước nốt ác tính để hệ thống ước lượng sát hơn.'
        : 'Kết quả đang ở vùng trung gian và chênh lệch giữa hai phía còn khá gần nhau.',
    supportingNote: null,
    advice,
    leaning,
  };
}
