type ViewKey = 'home' | 'predict' | 'history';

type Props = {
  activeView: ViewKey;
  historyCount: number;
  onNavigate: (view: ViewKey) => void;
};

export function Sidebar({ activeView, historyCount, onNavigate }: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="sidebar__mark">CT</span>
        <div>
          <strong>UngThuPhoi</strong>
        </div>
      </div>

      <nav className="sidebar__nav" aria-label="Điều hướng chính">
        <button
          type="button"
          className={`sidebar__link ${activeView === 'home' ? 'sidebar__link--active' : ''}`}
          onClick={() => onNavigate('home')}
        >
          <span>Trang chủ</span>
          <small>Giới thiệu hệ thống và quy trình xử lý</small>
        </button>

        <button
          type="button"
          className={`sidebar__link ${activeView === 'predict' ? 'sidebar__link--active' : ''}`}
          onClick={() => onNavigate('predict')}
        >
          <span>Dự đoán</span>
          <small>Ảnh CT và thông tin bệnh nhân</small>
        </button>

        <button
          type="button"
          className={`sidebar__link ${activeView === 'history' ? 'sidebar__link--active' : ''}`}
          onClick={() => onNavigate('history')}
        >
          <span>Lịch sử dự đoán</span>
          <small>{historyCount} bản ghi gần đây</small>
        </button>
      </nav>
    </aside>
  );
}
