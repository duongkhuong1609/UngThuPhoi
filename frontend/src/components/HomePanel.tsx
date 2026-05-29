export function HomePanel() {
  const processSteps = [
    'Người dùng chọn ảnh mẫu hoặc tải ảnh CT phổi',
    'Hệ thống kiểm tra ảnh đầu vào',
    'YOLO localization khoanh vùng nốt hoặc vùng nghi ngờ',
    'Model multimodal phân tích ảnh ROI và thông tin bệnh nhân',
    'Trả về mức nguy cơ, xác suất và kết luận hiện tại',
  ];

  return (
    <main className="home-page">
      <section className="panel home-intro-card">
        <div className="home-copy">
          <p className="eyebrow">Trang chủ</p>
          <h2>Hệ thống hỗ trợ dự đoán nguy cơ ác tính phổi</h2>
          <p className="home-copy__goal">
            <strong>Mục tiêu:</strong> hỗ trợ demo nghiên cứu theo hướng trực quan, nhất quán và dễ theo dõi.
          </p>
          <p className="home-copy__lead">
            Hệ thống hỗ trợ demo nghiên cứu dự đoán nguy cơ ác tính phổi từ ảnh CT và thông tin bệnh nhân,
            đồng thời cho phép xem vùng nghi ngờ được khoanh trong quá trình phân tích.
          </p>
        </div>
      </section>

      <section className="panel home-process-card">
        <div className="panel__header">
          <div>
            <h2>Quy trình xử lý</h2>
            <p className="panel__subtitle">Luồng xử lý chính của hệ thống từ lúc nạp ảnh đến khi trả về kết quả.</p>
          </div>
        </div>

        <div className="home-steps">
          {processSteps.map((step, index) => (
            <div key={step} className="home-step">
              <div className="home-step__index">{index + 1}</div>
              <div className="home-step__content">
                <strong>{step}</strong>
              </div>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
