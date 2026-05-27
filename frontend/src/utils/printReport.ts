function safeDocumentTitle(title: string): string {
  return title.replace(/[<>]/g, '').trim() || 'Bao cao du doan';
}

export function printElementById(elementId: string, title: string) {
  const element = document.getElementById(elementId);
  if (!element) {
    window.print();
    return;
  }

  const printWindow = window.open('', '_blank', 'width=980,height=720');
  if (!printWindow) {
    window.print();
    return;
  }

  const styles = Array.from(document.querySelectorAll('style, link[rel="stylesheet"]'))
    .map((node) => node.outerHTML)
    .join('\n');

  printWindow.document.write(`
    <!doctype html>
    <html lang="vi">
      <head>
        <meta charset="utf-8" />
        <title>${safeDocumentTitle(title)}</title>
        ${styles}
        <style>
          @page { size: A4; margin: 14mm; }
          body {
            margin: 0;
            background: #ffffff;
            color: #0f172a;
            font-family: Arial, sans-serif;
          }
          .print-wrapper {
            max-width: 920px;
            margin: 0 auto;
            padding: 18px;
          }
          .print-hidden,
          button {
            display: none !important;
          }
          img {
            max-width: 100%;
            break-inside: avoid;
          }
        </style>
      </head>
      <body>
        <main class="print-wrapper">
          ${element.outerHTML}
        </main>
      </body>
    </html>
  `);
  printWindow.document.close();
  printWindow.focus();

  window.setTimeout(() => {
    printWindow.print();
    printWindow.close();
  }, 350);
}
