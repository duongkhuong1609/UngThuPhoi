export function ModelPanel() {
  return (
    <main className="home-page">
      <section className="panel home-model-card">
        <div className="panel__header">
          <div>
            <p className="eyebrow">Giới thiệu model</p>
            <h2>Model hệ thống đang sử dụng</h2>
            <p className="panel__subtitle">Các thành phần model chính và vai trò của từng thành phần trong hệ thống hiện tại.</p>
          </div>
        </div>

        <div className="home-model-list">
          <div className="home-model-item">
            <span className="home-model-item__label">Classification model</span>
            <strong>
              Mô hình phân loại đa phương thức, kết hợp ảnh ROI và thông tin bệnh nhân để ước lượng nguy cơ thấp,
              trung gian hoặc cao.
            </strong>
          </div>

          <div className="home-model-item">
            <span className="home-model-item__label">YOLO localization model</span>
            <strong>
              Mô hình YOLO dùng để khoanh vùng nốt hoặc vùng nghi ngờ trên ảnh CT, giúp người dùng đối chiếu vị trí
              cần chú ý.
            </strong>
          </div>
        </div>
      </section>
    </main>
  );
}
