const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  PageBreak,
  PageNumber,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

// This script is stored with the geocoding code, while reports remain shared
// deliverables at the a_geo_tongji workspace root.
const ROOT = path.resolve(__dirname, "../..");
const OUT_COMBINED = path.join(ROOT, "output", "两种建筑高度估计方法综合报告");
const OUT_PAPER = path.join(ROOT, "output", "施展论文借鉴作用说明");
const TMP = path.join(ROOT, "tmp", "pdfs", "integrated_height_reports");
const FONT = "Noto Sans CJK SC";
const BLACK = "000000";
const PAGE_WIDTH = 11906;
const PAGE_HEIGHT = 16838;
const CONTENT_WIDTH = 9600;
const none = { style: BorderStyle.NONE, size: 0, color: BLACK };
const thin = { style: BorderStyle.SINGLE, size: 4, color: BLACK };
const thick = { style: BorderStyle.SINGLE, size: 8, color: BLACK };

const sourceImages = {
  pixelFeature: path.join(ROOT, "output/像素偏移建筑高度估计PICALL结果/独立PNG/08_合成孔径雷达建筑特征增强.png"),
  pixelRegistration: path.join(ROOT, "output/像素偏移建筑高度估计PICALL结果/独立PNG/15_数量质量联合配准.png"),
  pixelHeight: path.join(ROOT, "output/像素偏移建筑高度估计PICALL结果/独立PNG/16_数量质量联合建筑高度图.png"),
  psAssignment: path.join(ROOT, "output/同济校区PS三角面建筑估高方法报告/原始图片/基础结果/PNG/04_永久散射体屋顶墙面归属.png"),
  psMask: path.join(ROOT, "output/同济校区PS三角面建筑估高方法报告/原始图片/基础结果/PNG/10_局部幅度掩膜精化.png"),
  psFusion: path.join(ROOT, "output/同济校区PS三角面建筑估高方法报告/原始图片/高层优化/PNG/10_三角面投影与永久散射体融合.png"),
  psHeight: path.join(ROOT, "output/同济校区PS三角面建筑估高方法报告/原始图片/高层优化/PNG/09_高层优化验证.png"),
  paperProjection: path.join(TMP, "shizhan_page-15.png"),
  paperMask: path.join(TMP, "shizhan_page-16.png"),
};

for (const file of Object.values(sourceImages)) {
  if (!fs.existsSync(file)) throw new Error(`Missing source image: ${file}`);
}
fs.mkdirSync(OUT_COMBINED, { recursive: true });
fs.mkdirSync(OUT_PAPER, { recursive: true });
fs.mkdirSync(path.join(OUT_COMBINED, "引用原始图"), { recursive: true });
for (const key of ["pixelFeature", "pixelRegistration", "pixelHeight", "psAssignment", "psMask", "psFusion", "psHeight"]) {
  fs.copyFileSync(sourceImages[key], path.join(OUT_COMBINED, "引用原始图", path.basename(sourceImages[key])));
}
fs.copyFileSync(
  path.join(ROOT, "output/施展 附加轮廓矢量的SAR建筑物精细地理编码  初稿.pdf"),
  path.join(OUT_PAPER, "参考文献原文_施展_附加轮廓矢量的SAR建筑物精细地理编码_初稿.pdf"),
);
fs.copyFileSync(sourceImages.paperProjection, path.join(OUT_PAPER, "论文原文页_三维模型投影.png"));
fs.copyFileSync(sourceImages.paperMask, path.join(OUT_PAPER, "论文原文页_掩膜精炼.png"));

function run(text, options = {}) {
  return new TextRun({
    text: String(text),
    font: options.font || FONT,
    size: options.size || 21,
    color: BLACK,
    bold: Boolean(options.bold),
    italics: Boolean(options.italics),
  });
}

function para(text, options = {}) {
  return new Paragraph({
    alignment: options.alignment || AlignmentType.JUSTIFIED,
    spacing: { before: options.before || 0, after: options.after ?? 120, line: options.line || 340 },
    indent: options.indent === false ? undefined : { firstLine: 420 },
    keepNext: Boolean(options.keepNext),
    children: Array.isArray(text) ? text : [run(text, options)],
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    pageBreakBefore: true,
    spacing: { before: 0, after: 240 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLACK, space: 6 } },
    children: [run(text, { bold: true, size: 32 })],
  });
}

function h1Continue(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 300, after: 240 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLACK, space: 6 } },
    children: [run(text, { bold: true, size: 32 })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 140 },
    keepNext: true,
    children: [run(text, { bold: true, size: 26 })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 180, after: 100 },
    keepNext: true,
    children: [run(text, { bold: true, size: 23 })],
  });
}

function bullet(text, reference = "bullets") {
  return new Paragraph({
    numbering: { reference, level: 0 },
    spacing: { after: 80, line: 320 },
    children: [run(text)],
  });
}

function numbered(text, reference) {
  return new Paragraph({
    numbering: { reference, level: 0 },
    spacing: { after: 90, line: 320 },
    children: [run(text)],
  });
}

function equation(text, number) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 110, after: 150, line: 350 },
    keepNext: true,
    children: [run(text, { font: "Cambria Math", size: 21 }), run(`    (${number})`, { font: "Cambria Math", size: 20 })],
  });
}

function note(title, text) {
  return new Paragraph({
    spacing: { before: 130, after: 160, line: 330 },
    indent: { left: 260, right: 260 },
    border: {
      top: { style: BorderStyle.SINGLE, size: 4, color: BLACK, space: 6 },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: BLACK, space: 6 },
    },
    children: [run(`${title}：`, { bold: true }), run(text)],
  });
}

function tableCell(text, width, options = {}) {
  const borders = options.header
    ? { top: thick, bottom: thin, left: none, right: none }
    : options.last
      ? { top: none, bottom: thick, left: none, right: none }
      : { top: none, bottom: none, left: none, right: none };
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders,
    verticalAlign: VerticalAlign.CENTER,
    shading: { fill: "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 95, bottom: 95, left: 125, right: 125 },
    children: [
      new Paragraph({
        alignment: options.alignment || AlignmentType.LEFT,
        spacing: { after: 0, line: 280 },
        children: [run(text, { bold: Boolean(options.header), size: 19 })],
      }),
    ],
  });
}

function table(headers, rows, widths) {
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true,
        cantSplit: true,
        children: headers.map((x, i) => tableCell(x, widths[i], { header: true })),
      }),
      ...rows.map((row, r) =>
        new TableRow({
          cantSplit: true,
          children: row.map((x, i) => tableCell(x, widths[i], { last: r === rows.length - 1 })),
        }),
      ),
    ],
  });
}

function pngSize(file) {
  const data = fs.readFileSync(file);
  return { width: data.readUInt32BE(16), height: data.readUInt32BE(20) };
}

function figurePage(file, caption, explanation, options = {}) {
  const size = pngSize(file);
  const maxWidth = options.maxWidth || 640;
  const maxHeight = options.maxHeight || 650;
  const scale = Math.min(maxWidth / size.width, maxHeight / size.height, 1);
  return [
    new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_2, children: [run(caption, { bold: true, size: 26 })] }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 100 },
      children: [
        new ImageRun({
          type: "png",
          data: fs.readFileSync(file),
          transformation: { width: Math.round(size.width * scale), height: Math.round(size.height * scale) },
          altText: { title: caption, description: caption, name: path.basename(file) },
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 140, line: 290 },
      children: [run(caption, { size: 18 })],
    }),
    para(explanation),
  ];
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function docOptions(title, children) {
  return new Document({
    creator: "Codex",
    title,
    description: title,
    styles: {
      default: { document: { run: { font: FONT, size: 21, color: BLACK } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 32, bold: true, color: BLACK },
          paragraph: { spacing: { before: 0, after: 240 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 26, bold: true, color: BLACK },
          paragraph: { spacing: { before: 240, after: 140 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 23, bold: true, color: BLACK },
          paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 2 } },
      ],
    },
    numbering: {
      config: [
        { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
        { reference: "combined-steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
        { reference: "pixel-opt-steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
        { reference: "pixel-strict-steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
        { reference: "combined-joint", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
        { reference: "paper-steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      ],
    },
    sections: [{
      properties: {
        page: { size: { width: PAGE_WIDTH, height: PAGE_HEIGHT }, margin: { top: 1080, right: 1153, bottom: 1080, left: 1153 } },
      },
      headers: {
        default: new Header({ children: [new Paragraph({
          spacing: { after: 0 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLACK, space: 4 } },
          children: [run(title, { size: 16 })],
        })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [run("第 ", { size: 16 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: BLACK }), run(" 页", { size: 16 })],
        })] }),
      },
      children,
    }],
  });
}

function cover(title, subtitle, sourceLine) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1050, after: 300 }, children: [run(title, { bold: true, size: 48 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 720 }, children: [run(subtitle, { size: 25 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [run(sourceLine, { size: 20 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1800 }, children: [run("同济校区SAR建筑物研究整理", { size: 20 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160 }, children: [run("2026年7月", { size: 20 })] }),
    pageBreak(),
  ];
}

async function buildCombined() {
  const c = [];
  c.push(...cover("建筑高度估计两种方法综合报告", "像素偏移估高与PS三角面估高", "依据现有PICALL成果、PS三角面成果与施展论文初稿整理"));
  c.push(
    h1("先说清楚：两种方法究竟在做什么"),
    para("当前有两条相互独立的建筑高度估计路线。它们都使用建筑轮廓、SAR成像几何和4 m统一地面基底，但“拿什么来量高度”完全不同。先把这一区别弄清，后面的公式就容易理解。"),
    h2("方法一：像素偏移法——看屋顶在SAR图上移动了多少"),
    para("先把建筑屋顶按一个候选高度投影到SAR图上，再与真实影像中的屋顶边缘、连续亮线和内外强度变化比较。如果投影轮廓偏在一边，就改变候选高度并重新投影。哪个高度使投影轮廓与影像结构最吻合，就把哪个高度作为结果。"),
    para("可以把它想成一个可上下移动的透明建筑模型：模型升高时，侧视SAR中的屋顶投影会沿特定方向移动；不断调节模型高度，直到透明屋顶与影像中的屋顶结构对齐。这里真正的观测是“二维像素位置”，高度是通过严格投影反求出来的。"),
    h2("方法二：PS三角面法——把楼身不同位置的PS当成多把高度尺"),
    para("PS点带有高程和相干性，但并非每个点都在楼顶。有的点来自屋顶，有的来自墙面中部。方法先把建筑做成屋顶和墙面的三角网，投影到SAR图上，判断每个PS落在哪个三角面，并计算它位于整栋楼高度的比例。例如屋顶点比例为1，墙面中部点可能为0.5或0.6。"),
    para("随后把许多PS写成高度方程共同求解。一条墙面PS观测到18 m、且位于楼身60%处时，它提供的不是“建筑高18 m”，而是“建筑高度的60%约为18 m”。多条质量不同的PS方程经过稳健平差，得到整栋建筑高度。"),
    note("最重要的共同口径", "两条路线都采用4.000 m统一地面基底。屋顶绝对高程减去4 m才是建筑离地高度。现有质量等级反映内部图像与几何可靠性，不等同于独立LiDAR、GNSS或实测高度精度。"),
    table(
      ["处理环节", "像素偏移法", "PS三角面法"],
      [
        ["输入", "三景RSLC、建筑屋顶轮廓、轨道参数", "PS位置与高程、建筑三角网、SAR影像"],
        ["直接观测", "投影屋顶与SAR结构之间的二维偏移", "PS离地高程及其屋顶/墙面比例"],
        ["核心计算", "候选高度重投影+形状匹配", "逐PS观测方程+Huber稳健平差"],
        ["输出", "屋顶绝对高程Z及离地高度H", "建筑高度H及墙面共同偏差β"],
        ["当前覆盖", "369 / 1028栋", "324栋"],
        ["不能混淆", "图像位移不是直接的米制高度", "墙面PS高程不是屋顶高度"],
      ],
      [1900, 3850, 3850],
    ),
    note("两条路线的关系", "像素偏移法从“位置”反演高度，PS三角面法从“高程和表面比例”平差高度。二者是两条独立证据链，不是同一算法的两个步骤。"),
  );

  c.push(
    h1("1 建筑高度为什么会表现为像素偏移"),
    h2("1.1 先看侧视几何"),
    para("普通照片接近从上向下看，屋顶大致落在建筑底部轮廓内部；SAR从斜上方照射，同一个平面位置上，屋顶比地面更靠近卫星，所以回波更早到达，屋顶会向近距方向移动。建筑越高，这个位移一般越大。这就是用像素位置估计高度的几何依据。"),
    equation("Δx_ground ≈ H / tan θ", "1"),
    para("θ为从竖直方向量起的局部入射角，H为建筑离地高度，Δx_ground是屋顶相对底部在地距方向上的近似位移。该式适合直观理解：入射角越小，单位高度造成的地面投影位移越大。"),
    equation("ΔR ≈ -H cos θ,    Δc ≈ ΔR / δR_pixel", "2"),
    para("在原始斜距影像中，更直接的是斜距变化ΔR。δR_pixel是斜距像元间隔，Δc是距离向列偏移；负号表示屋顶升高后斜距变短，向近距侧移动。地距公式和斜距公式描述的是同一现象，只是坐标系不同，不能把两者的像元间隔混用。"),
    h3("一个仅用于理解的数值例子"),
    para("假设入射角θ=35°，建筑高30 m。地距位移约为30/tan35°=42.8 m；若地距像元为1.5 m，约移动28.6像素。若在斜距坐标中像元间隔为0.9 m，则斜距变化约为-30cos35°=-24.6 m，对应-27.3像素。数值接近，但这是示意计算，不是当前单栋建筑的正式高度结果。"),
    h2("1.2 为什么正式处理不能只用“偏移像素×固定比例”"),
    bullet("建筑轮廓中每个顶点的卫星视线方向不完全相同，局部入射角会变化。"),
    bullet("SAR行列坐标是二维的，高度主要影响距离向，但轨道曲率、坐标转换和建筑方向也会带来方位向分量。"),
    bullet("全局轨道或数据解析误差、局部配准误差也会造成像素偏移，若全部当成高度，会产生系统性高估或低估。"),
    bullet("真实亮线可能来自屋顶边缘、墙地二面角、邻楼或金属结构，必须同时检查位置、形状和多景一致性。"),
    para("因此，近似公式只用于解释规律和生成搜索初值；最终结果对每个候选高程都重新执行严格距离-多普勒投影。"),
    h2("1.3 严格投影方程"),
    equation("R(t) = || X - S(t) ||", "3"),
    equation("(X - S(t)) · V(t) = 0", "4"),
    para("X是建筑三维点，S(t)和V(t)是卫星在时刻t的位置与速度。第一式决定斜距位置，第二式决定零多普勒成像时刻。求得t和R后，再根据PRF、起始斜距和像元间隔换成方位行与距离列。建筑高度改变时，X改变，t、R和二维像素位置都要重新计算。"),
    h2("1.4 统一高程基准"),
    equation("H_building = Z_roof - Z_ground = Z_roof - 4.000 m", "5"),
    para("严格投影搜索的变量是屋顶绝对高程Z_roof，最终成果需要的是离地高度H_building。当前全区统一令模型地面绝对高程为4.000 m，所以必须在求得屋顶绝对高程后减去4 m。4 m不是每栋建筑的实测地面，后续仍应升级为逐建筑地面高程。"),
  );

  c.push(
    h1Continue("2 方法一：像素偏移建筑高度估计"),
    h2("2.1 从二维像素偏移得到高度初值"),
    para("先对同一栋建筑做两次严格投影：一次用模型地面高程，另一次用一个已知的候选屋顶高程。两次投影位置之差给出这栋建筑在当前成像几何下的“每米高度对应二维像素移动方向”。"),
    equation("g_b ≈ [ p_b(Z_1) - p_b(Z_0) ] / (Z_1 - Z_0)", "6"),
    para("p_b(Z)可取屋顶投影轮廓的中心或同一组顶点的二维坐标，g_b=[g_row,g_col]的单位是像素/米。它不是全区统一常数，而是逐建筑、逐场景由严格投影得到。"),
    para("图像搜索得到建筑轮廓的二维改正量δp=[δrow,δcol]后，把它投影到高度敏感方向g_b上，可得到线性高度改正初值："),
    equation("δH_linear = (g_b · δp) / (g_b · g_b)", "7"),
    equation("δp_perp = δp - g_b δH_linear", "8"),
    para("δH_linear是偏移中可以由高度解释的部分；δp_perp与高度方向垂直，主要反映局部配准误差、轮廓平面误差或图像结构误匹配。这样可以避免把所有row/col偏移都硬解释成楼高。"),
    h3("简化例子"),
    para("假设严格投影得到g_b=[0.05,-0.80]像素/米，图像匹配改正量为δp=[1.0,-16.0]像素，则δH_linear约为20 m。若直接只看总位移16.03像素并乘某个全区常数，可能把1像素的方位向配准误差也算入高度；二维投影分解能把这部分单独保留。"),
    h2("2.2 投影优化为什么要分层"),
    para("影像中的偏移并不只有高度造成。当前流程把投影优化拆成“全局改正—逐建筑图像配准—高度/横向分解—严格高程重投影—多景质量控制”五个层次，每一层只解决自己能解释的问题。"),
    numbered("三景共注册：三景RSLC已经在同一雷达坐标网格中，先消除景与景之间的大尺度错位。", "pixel-opt-steps"),
    numbered("全局雷达改正：统一应用row +34、col -1，消除轨道、时间或数据解析造成的整体偏移。这个改正不能被当成建筑高度。", "pixel-opt-steps"),
    numbered("SAR特征增强：对数压缩、保边去斑、局部对比和多尺度边缘融合，使屋顶边缘与连续亮线更容易被评分。增强结果只用于定位。", "pixel-opt-steps"),
    numbered("形态自适应二维搜索：根据建筑面积、长宽比和方向设置窗口；先以2像素步长粗搜，再以0.25像素步长细搜，得到δrow和δcol。", "pixel-opt-steps"),
    numbered("高度方向分解：用式(7)得到高度改正初值，用式(8)保留垂直于高度方向的局部残差。", "pixel-opt-steps"),
    numbered("空间残差场：把邻近可靠建筑的横向残差形成平滑局部改正，当前限制在±3像素，避免残差场吞掉真实的高度位移。", "pixel-opt-steps"),
    numbered("三景联合：三景分别定位，优先保留彼此接近的场景对，再用中位数融合，降低单景散斑或遮挡的影响。", "pixel-opt-steps"),
    numbered("严格高程搜索：围绕高度初值逐个改变屋顶绝对高程，每个候选都重新做距离-多普勒投影，最后由位置与形状误差选优。", "pixel-opt-steps"),
    h2("2.3 图像匹配怎样判断“对齐了”"),
    equation("S = w_e E_edge + w_c E_cont + w_i C_in/out + w_b B_bright", "9"),
    para("E_edge衡量投影边界是否落在明显边缘上；E_cont衡量边缘是否连续；C_in/out比较轮廓内外的强度差；B_bright检查目标附近是否存在合理亮散射。不同项共同限制，避免只追逐单个最亮像素。"),
    para("只用亮度会被强散射点吸引，只用中心距离又可能让形状不对的轮廓通过。因此评分同时看边缘、连续性、内外对比和亮散射，并对搜索到边界、形状断裂或位移异常的候选降级。"),
    h2("2.4 从高度初值到最终严格高度"),
    numbered("以线性高度改正为中心，先按1 m步长搜索较宽高程范围。", "pixel-strict-steps"),
    numbered("对每个候选Z重新生成屋顶三维点，并重新执行式(3)—式(4)的距离-多普勒投影。", "pixel-strict-steps"),
    numbered("计算投影屋顶与图像候选的中心距离和Hausdorff边界距离。", "pixel-strict-steps"),
    numbered("在粗搜索最优值附近按0.1 m步长细搜，得到屋顶绝对高程。", "pixel-strict-steps"),
    numbered("减去4 m模型地面，转换为建筑离地高度，并执行三景一致性和残差阈值检查。", "pixel-strict-steps"),
    equation("J(Z) = d_centroid[Π_b(Z),Q_b] + 0.15 d_Hausdorff[Π_b(Z),Q_b]", "10"),
    para("Π_b(Z)是候选高程Z下的严格投影屋顶，Q_b是从SAR影像定位的屋顶结构。中心项防止整体跑偏，Hausdorff项约束边界形状。J(Z)最小的候选只是数值最优，还必须通过场景一致性和最大残差门槛。"),
    h2("2.5 置信度与拒绝规则"),
    table(
      ["等级", "主要内部条件", "解释"],
      [
        ["高", "J≤2像素，最近场景对≤2.5像素", "位置与多景一致性最好"],
        ["中", "综合残差≤4像素", "可用，但需保留质量字段"],
        ["低", "综合残差≤7像素", "仅用于谨慎分析"],
        ["拒绝", "超过门槛或搜索不稳定", "高度保持空值，不以先验填充"],
      ],
      [1400, 4100, 4100],
    ),
    h2("2.6 当前结果"),
    para("当前最终结果共1028栋建筑，其中369栋获得有效高度，659栋保持无值；高、中、低、补充级分别为106、37、34、192栋。严格来源185栋，可靠混合补充184栋。有效高度均值20.47 m，中位数17.10 m，范围3.30–98.61 m。"),
    para([run("覆盖口径：", { bold: true }), run("369栋表示在当前阈值下通过图像匹配和几何一致性检查的覆盖数量，不代表这些建筑都达到某个外部实测误差，也不代表其余659栋高度为0。")], { indent: false }),
  );
  c.push(...figurePage(sourceImages.pixelFeature, "图1  三景SAR建筑特征增强", "图像增强把屋顶边缘、墙角亮线和局部对比变得更清楚，但增强结果只参与匹配评分，不直接产生高度。"));
  c.push(...figurePage(sourceImages.pixelRegistration, "图2  数量—质量联合优化配准", "该图展示最终推荐的建筑局部配准。投影轮廓与影像结构的接近程度决定候选高度能否进入严格搜索与质量筛选。"));
  c.push(...figurePage(sourceImages.pixelHeight, "图3  像素偏移法最终建筑高度图", "有色建筑为369栋有效估高结果，灰色建筑保持无值。地图显示的是当前统一4 m地面基底下的离地高度。"));

  c.push(
    h1("3 方法二：PS三角面建筑高度估计"),
    h2("3.1 原理"),
    para("PS点带有空间位置、高程和相干性，但一个PS可能来自屋顶、墙面、墙角或邻近目标。若直接对PS高程求平均，会把墙面点当成屋顶点，也可能把邻楼亮点混入。因此先将建筑构造成屋顶、墙面和底面的三角网，严格投影到SAR图上，再判断PS属于哪个三角面。"),
    h2("3.2 三角面与重心坐标"),
    equation("p_i = α_i v_1 + β_i v_2 + γ_i v_3,   α_i + β_i + γ_i = 1", "11"),
    para("PS落入投影三角形后，由三个重心权重描述其表面位置。屋顶三角形三个顶点均在顶部，因此垂直比例f_i=1；墙面三角形含顶部和底部顶点，属于顶部顶点的权重之和就是f_i。比如f_i=0.6，表示PS位于楼身约60%的高度处。"),
    h2("3.3 掩膜精化"),
    para("几何投影只是候选范围，内部仍可能包含道路、树木或邻楼散射。当前流程借鉴施展论文的局部强度思想，在投影外侧估计背景，再用亮度阈值、形态学闭运算和连通域筛选收紧支持区，且最终掩膜始终限制在原三角面内部。"),
    equation("τ = μ_bg + κ σ_bg", "12"),
    equation("M_refined = M_geometry ∩ { A(r,c) > τ } ∩ M_connected", "13"),
    para("μ_bg和σ_bg来自建筑附近的局部背景，κ控制筛选严格程度。局部阈值比全图统一阈值更能适应不同建筑、不同立面和不同散射强度。几何范围负责“不串楼”，强度与连通性负责“去弱背景、留主体”。"),
    h2("3.4 逐PS高度方程与稳健平差"),
    equation("y_i = z_i - 4 = f_i H_b + I_wall,i β_b + ε_i", "14"),
    para("H_b是待求建筑高度，β_b吸收墙面散射相对理想线性高度模型的共同偏差。屋顶PS的f_i=1，直接约束整栋楼；墙面PS按其垂直比例参与。一个绝对高程22 m、f_i=0.6的墙面PS，其离地观测为18 m，约束式近似为18=0.6H_b+β_b，而不是把18 m当成楼顶。"),
    equation("min Σ_i w_i ρ_δ(y_i - f_i H_b - I_wall,i β_b) + λ β_b²", "15"),
    para("w_i综合相干性、三角形内部程度、垂直杠杆、归属歧义和配准可靠性。Huber损失对小残差按平方处理，对大残差近似线性处理，既保留多数有效PS，也减弱异常点的支配作用。"),
    h2("3.5 高楼顶部恢复"),
    para("高层建筑常缺少真正位于屋顶顶部的稳定PS，普通稳健中心容易系统性偏低。当前高层分支对先验不低于30 m、基础解通过质量控制且至少有3个可靠PS的建筑，读取PS校正高度第95百分位，并用非对称Huber-ridge校准与随高度增加的恢复下限修正。第95百分位反映稳定的高端证据，比最大值更不易被单个异常点抬高。"),
    equation("H_high = max{ H_calibrated(q95, H_center, …),  L(H_prior) }", "16"),
    note("为什么仍需谨慎", "H_prior同时参与初始三维投影与高层恢复下限，所以高层误差只能解释为与现有Shapefile几何先验的一致性，不能写成独立真实精度。"),
    h2("3.6 闭环重投影"),
    para("高度更新后，屋顶和墙面在SAR图中的位置也会变化。因此流程必须重新建模、重新投影、重新精化掩膜、重新分配PS并再次平差，直到几何位置与高度结果一致。不能只改最终表格中的高度数值。"),
    h2("3.7 当前结果"),
    para("当前输入23,178个PS点，最终13,432个获得建筑表面归属；324栋建筑保留有效高度，其中279栋为高/中质量主结果，45栋为PS支撑补充级，34栋应用高层顶部恢复。36栋高层相对Shapefile几何先验的MAE由27.83 m降至5.85 m，RMSE由30.52 m降至7.37 m，最大欠估由54.22 m降至7.85 m。"),
  );
  c.push(...figurePage(sourceImages.psAssignment, "图4  PS点的屋顶与墙面三角形归属", "这一步把“一个PS在哪栋楼附近”进一步细化为“它落在屋顶还是墙面，以及在墙面高度的哪个比例”。"));
  c.push(...figurePage(sourceImages.psMask, "图5  局部幅度掩膜精化", "局部背景阈值、形态学与连通域共同收紧几何候选区；精化不会允许掩膜越过原建筑三角面去吸收邻楼强散射。"));
  c.push(...figurePage(sourceImages.psFusion, "图6  全区域三角面投影与PS融合", "该图保留全区域和典型建筑局部放大，用于检查最终高度变化后PS表面归属是否仍与严格投影一致。"));
  c.push(...figurePage(sourceImages.psHeight, "图7  PS三角面法高层优化与正式高度结果", "正式高度成果以结果目录图09为准。高层顶部恢复强调不因稳健求解而保守压低，但仍保留质量门槛和内部一致性审计。"));

  c.push(
    h1("4 两种方法如何选择与联合使用"),
    table(
      ["比较项", "像素偏移法", "PS三角面法"],
      [
        ["最直观的证据", "屋顶轮廓在SAR图中的位置", "PS点相对地面的高程"],
        ["投影的作用", "每个候选高度都重投影并评分", "先确定PS对应建筑表面和比例"],
        ["适合场景", "边缘清楚、三景一致、PS稀少", "PS丰富、相干性高、墙面/屋顶可分"],
        ["高楼处理", "扩大严格高程搜索并检查多景一致性", "读取可靠PS上尾并闭环重投影"],
        ["缺失处理", "不以先验、均值或邻域高度填充", "补充级仍必须有PS方程支撑"],
        ["当前覆盖", "369栋", "324栋"],
      ],
      [1900, 3850, 3850],
    ),
    h2("4.1 推荐联合策略"),
    numbered("先分别保留两套方法的独立结果、质量等级和来源字段，不直接把高度平均。", "combined-joint"),
    numbered("在两套方法都有结果的建筑上计算差值d_b=H_PS,b-H_pixel,b，按建筑高度层级和形态检查系统偏差。", "combined-joint"),
    numbered("用差值中位数和MAD识别异常：|d_b-median(d)|较大时，回看影像配准、PS表面归属、局部掩膜与基底高程。", "combined-joint"),
    numbered("只有在两套结果质量都可靠、观测误差近似独立且偏差已校正时，才考虑按不确定度加权融合；否则保留双结果比盲目平均更可审计。", "combined-joint"),
    equation("H_fused = (H_pixel/σ_pixel² + H_PS/σ_PS²) / (1/σ_pixel² + 1/σ_PS²)", "17"),
    para("式(17)只是满足条件时的融合形式。若σ只是内部评分而非可校准的不确定度，就不应套用该式生成“更精确”的高度。"),
    h2("4.2 两种方法互相检查什么"),
    bullet("像素偏移高、PS三角面低：检查高楼顶部PS缺失、PS上尾证据是否被稳健中心压低。"),
    bullet("PS三角面高、像素偏移低：检查图像搜索是否锁定墙角或低层裙房，而非主楼屋顶。"),
    bullet("两者都偏向先验：检查Shapefile高度是否同时影响起点、搜索范围和高层恢复下限。"),
    bullet("两者差异随空间变化：检查全局row/col改正是否不足，以及局部垂直残差场是否吸收了真实高度位移。"),
  );

  c.push(
    h1("5 施展论文在两种方法中的作用"),
    para("施展《附加轮廓矢量的SAR建筑物精细地理编码》提供的是“前向几何建模与建筑散射点精细定位”框架：用轮廓和已知高度构建三维挤出模型，按零多普勒几何投影，生成初始掩膜，再利用局部SAR幅度精炼掩膜，并通过三角面重心坐标把像素映射回建筑表面。"),
    table(
      ["论文思想", "当前工作的借鉴", "当前工作的扩展"],
      [
        ["严格三维模型投影", "两种估高方法共用几何基础", "把高度从已知输入改为待搜索或待平差未知量"],
        ["初始掩膜+局部强度精炼", "用于PS三角面表面支持区收紧", "加入不越过原三角面的约束和多项质量检查"],
        ["三角面重心反算", "用于PS屋顶/墙面归属和垂直比例", "由位置插值扩展为逐PS建筑高度观测方程"],
        ["LOS前景归属", "为密集高楼重叠像素处理提供思路", "尚可进一步发展为全区建筑间联合竞争"],
      ],
      [2300, 3400, 3900],
    ),
    note("明确结论", "该论文没有提出“在建筑高度未知时，由SAR观测反演建筑高度”的独立估高方法。论文把楼层数×平均层高或OSM高度作为先验输入，用于构建模型；输出重点是精炼掩膜和建筑表面三维散射点。当前两种估高方法是在其前向投影框架上做的反问题扩展。"),
  );

  c.push(
    h1("6 局限性与结论"),
    h2("6.1 当前局限"),
    bullet("统一4 m地面无法表达逐建筑真实基底高程；应优先接入DSM或建筑周边稳健地面估计。"),
    bullet("像素偏移的图像边缘可能由墙地二面角、邻楼、阴影或强散斑产生，三景一致性只能降低风险，不能替代实测验证。"),
    bullet("PS三角面法依赖PS高程质量、表面归属和三角网几何；顶部PS缺失仍是高楼估计的核心难点。"),
    bullet("Shapefile高度参与几何建模与部分高层恢复，内部先验一致性不可写成真实高度精度。"),
    bullet("两套方法尚未使用同一批独立LiDAR或实测楼高做全区外部验证。"),
    h2("6.2 结论"),
    para("像素偏移法从“屋顶在影像中的位置”反演高度，PS三角面法从“PS在建筑表面上的高程与比例”平差高度。二者共享严格距离-多普勒投影和4 m地面基准，但观测来源、误差模型和质量等级不同。当前结果应作为两条独立证据链保存，并在重叠建筑上做差异诊断；只有完成不确定度校准和外部验证后，才适合形成统一融合高度。"),
    h2("6.3 数据来源说明"),
    para("本报告依据两个现有成果包及施展论文初稿整理：output/同济校区PS三角面建筑估高方法报告、output/像素偏移建筑高度估计PICALL结果、output/施展 附加轮廓矢量的SAR建筑物精细地理编码  初稿.pdf。报告未调用外部网络资料。"),
  );

  const file = path.join(OUT_COMBINED, "建筑高度估计两种方法综合报告.docx");
  fs.writeFileSync(file, await Packer.toBuffer(docOptions("建筑高度估计两种方法综合报告", c)));
  return file;
}

async function buildPaper() {
  const c = [];
  c.push(...cover("施展论文借鉴作用说明", "与建筑高度估计方法的边界辨析", "文献：施展《附加轮廓矢量的SAR建筑物精细地理编码》（初稿，2026）"));
  c.push(
    h1("摘要结论"),
    note("结论先行", "这篇论文有“建筑高度的使用方法”，但没有“建筑高度的估计方法”。它把楼层数换算高度或OSM高度当作已知先验，构建三维建筑模型并完成SAR精细地理编码；没有把高度设为未知量，也没有通过像素偏移、相位、高程观测或优化目标反演建筑高度。"),
    para("论文真正解决的问题是：已知建筑平面轮廓和大致高度后，怎样把三维建筑正确投影到SAR雷达坐标系，怎样用影像强度收紧掩膜，怎样把散射像素反算到建筑表面，并在相邻高楼叠掩时判断像素属于哪栋建筑。"),
    table(
      ["判断项", "论文是否具备", "依据"],
      [
        ["使用建筑高度", "有", "楼层数×3 m得到105 m；陆家嘴建筑使用216、216、230 m先验"],
        ["估计未知建筑高度", "没有", "高度在三维建模前已经给定，不是待求参数"],
        ["建筑三维散射点定位", "有", "精炼像素通过三角面重心坐标映射回建筑表面"],
        ["投影与掩膜精炼", "有", "距离-多普勒投影、局部强度阈值、连通与形态学约束"],
        ["独立高度精度验证", "没有", "精度指标主要评价散射点到建筑轮廓边界的定位距离"],
      ],
      [2200, 1900, 5500],
    ),
  );

  c.push(
    h1("1 文献研究目标与技术路线"),
    para("论文题为《附加轮廓矢量的SAR建筑物精细地理编码》，作者施展，2026年同济大学测绘工程本科毕业设计初稿。研究目标不是输出每栋楼的未知高度，而是提高城市高层建筑散射点的地理编码可靠性。"),
    h2("1.1 技术路线"),
    numbered("读取建筑物轮廓和高度先验。", "paper-steps"),
    numbered("将二维轮廓向上挤出，构建屋顶、墙面和底面的三维三角网。", "paper-steps"),
    numbered("利用SAR斜距方程和零多普勒方程，把三角面顶点投影到雷达坐标。", "paper-steps"),
    numbered("栅格化投影三角面，得到建筑物初始掩膜。", "paper-steps"),
    numbered("在局部窗口内依据SAR幅度、连通性和形态学约束精炼掩膜。", "paper-steps"),
    numbered("把精炼像素通过三角面重心坐标反算为WGS84/ECEF三维散射点。", "paper-steps"),
    numbered("在高楼叠掩区域依据LOS前景顺序分配重叠像素。", "paper-steps"),
    h2("1.2 论文的核心输出"),
    para("核心输出是与建筑表面几何一致的散射点集合及三维点云，而不是一张由SAR反演得到的建筑高度图。论文以点到建筑物轮廓边界的水平距离评价定位结果，这与建筑高度误差MAE/RMSE是不同指标。"),
  );

  c.push(
    h1("2 论文到底有没有建筑高度估计"),
    h2("2.1 有：高度先验的构造和使用"),
    equation("H_b = N_f h_f", "1"),
    para("论文第2.3节把楼层数N_f乘以平均层高h_f得到建筑高度H_b。四栋住宅楼按35层、每层3 m估算为105 m；陆家嘴B11、B8、B2使用约216 m、216 m和230 m的既有高度。这些高度随后用于生成顶部顶点。"),
    equation("T_i = LLH2ECEF(λ_i, φ_i, h_g + H_b)", "2"),
    para("式(2)说明高度在投影之前已经确定，用它把底部轮廓抬升为顶部轮廓。这里的“估算”是楼层数乘经验层高或读取外部属性，不是从SAR影像观测中反演。"),
    h2("2.2 没有：把高度设为SAR反演未知量"),
    para("若论文具有建筑估高方法，通常应明确把H_b列为未知量，并给出由SAR观测驱动的目标函数或观测方程，例如最小化投影屋顶与图像边缘的距离，或用PS高程和墙面比例平差H_b。原文没有这类求解链。"),
    note("判别标准", "“高度影响投影”“用高度建模”“楼层数乘层高”都不等于“由SAR估计高度”。前者是前向模型或外部先验；后者必须从未知高度出发，由SAR观测约束并输出高度及其质量。"),
    h2("2.3 原文对这一边界的直接提示"),
    bullet("引言明确把问题限定为“在已有建筑物轮廓矢量和高度先验的条件下”提升散射点定位。"),
    bullet("方法总体流程第一步就是读取轮廓与高度先验，随后才构建三维模型。"),
    bullet("误差来源指出高度来自楼层数估算，真实高度可能与模型不一致。"),
    bullet("展望提出未来接入更精细的建筑高度、激光雷达或城市三维模型，说明当前高度不是SAR反演结果。"),
  );

  c.push(...figurePage(sourceImages.paperProjection, "图1  论文原文：三维模型雷达坐标投影", "该页说明模型顶点由已知经纬高转换到ECEF，再用相同距离-多普勒几何投影。它提供了可靠的前向投影框架，但高度在进入投影前已经给定。", { maxHeight: 690, maxWidth: 500 }));
  c.push(...figurePage(sourceImages.paperMask, "图2  论文原文：局部强度掩膜精炼", "该页给出τ=μ_A+κσ_A和M_f=M_0∩{A>τ}，并强调强度只负责保留真实散射、剔除弱背景，几何投影仍是主约束。", { maxHeight: 690, maxWidth: 500 }));

  c.push(
    h1("3 对当前建筑估高工作的具体借鉴"),
    h2("3.1 严格距离-多普勒投影"),
    equation("R(t) = ||X-S(t)||,   (X-S(t))·V(t)=0", "3"),
    para("论文把建筑三维点投影到雷达行列坐标，为当前两种估高方法提供共同的前向模型。像素偏移法把这一模型反复用于不同候选高度；PS三角面法用它判断PS落在哪个建筑表面。"),
    h2("3.2 三角面建模与重心坐标"),
    equation("p = αV_1 + βV_2 + γV_3,   α+β+γ=1", "4"),
    para("论文用三角面把二维像素反算到三维建筑表面。当前PS三角面估高进一步把顶部顶点的重心权重转成垂直比例f_i，使墙面PS可以按其所在高度比例约束整栋楼。"),
    h2("3.3 掩膜精炼"),
    equation("τ = μ_A + κσ_A,   M_f = M_0 ∩ {A(r,c)>τ}", "5"),
    para("论文强调在初始几何掩膜附近做局部强度筛选，而不是全图阈值分割。这一思想被用于当前PS流程：在建筑投影附近估计背景，并增加形态学闭运算、连通域筛选和“不越过原三角面”约束，减少道路、树木和邻楼亮点。"),
    h2("3.4 叠掩区域LOS归属"),
    para("论文对多个建筑投影掩膜的重叠像素按近距侧列号分配给更靠近传感器的前景建筑。这为密集高楼的建筑间竞争提供了可解释规则，也提示当前全区流程未来应从逐建筑局部处理升级为联合归属。"),
  );

  c.push(
    h1("4 当前两种估高方法如何超出论文"),
    table(
      ["内容", "施展论文", "当前像素偏移法", "当前PS三角面法"],
      [
        ["高度角色", "已知先验输入", "待搜索未知量", "待平差未知量"],
        ["SAR观测", "幅度用于掩膜", "屋顶边缘与二维偏移", "PS高程、相干性与表面比例"],
        ["高度方程", "无", "min J(Z)", "y_i=f_iH_b+I_wallβ_b+ε_i"],
        ["高楼优化", "无估高分支", "严格扩大搜索与多景一致性", "PS第95百分位与顶部恢复"],
        ["主要输出", "建筑表面散射点云", "369栋离地高度", "324栋离地高度"],
      ],
      [1700, 2350, 2700, 2850],
    ),
    h2("4.1 从前向模型到反问题"),
    para("论文回答“给定高度H，建筑会投影到哪里”。当前像素偏移法把这个问题反过来：寻找哪个H使投影最贴近SAR图像。当前PS三角面法则利用投影确定PS的表面比例f_i，再通过多条高度观测方程求H。"),
    equation("前向：H → Π(H)；  反演：H* = arg min_H D[Π(H), SAR]", "6"),
    equation("PS反演：H* = arg min_H,β Σ_i w_i ρ(y_i-f_iH-I_wall,iβ)", "7"),
    note("引用时建议", "可写“当前方法借鉴了施展论文的三维建筑投影、局部掩膜精炼和三角面反算框架，并进一步把建筑高度由已知输入改为待搜索或待平差参数”。不要写“施展论文提出了建筑高度反演方法”。"),
  );

  c.push(
    h1("5 使用边界、风险与结论"),
    h2("5.1 可以借鉴的结论"),
    bullet("轮廓与高度先验能把散射点从单一DEM地表约束扩展到建筑三维表面。"),
    bullet("局部掩膜精炼应以几何投影为主约束，影像强度负责筛除背景，而不是无限扩张边界。"),
    bullet("三角面重心坐标适合建立雷达像素与三维建筑表面之间的连续映射。"),
    bullet("密集高楼叠掩需要建筑间归属规则，不能让同一像素被多栋楼重复解释。"),
    h2("5.2 不能直接借用的结论"),
    bullet("论文的建筑高度值来自楼层数或外部数据，不能作为SAR反演高度有效性的证据。"),
    bullet("点到轮廓边界的定位误差不能替代建筑高度MAE、RMSE或绝对高程精度。"),
    bullet("在高度参与建模后再用同一高度先验评价一致性，会产生循环依赖；应明确称为内部几何一致性。"),
    h2("5.3 最终结论"),
    para("施展论文对当前工作最重要的作用，是建立了可信的三维建筑前向投影、局部掩膜精炼、三角面反算和LOS叠掩归属框架。它没有提供SAR建筑高度反演方法。当前像素偏移法和PS三角面法是在该框架上分别引入图像位置观测与PS高程观测，把已知高度的前向地理编码问题扩展为未知高度的反问题。"),
    h2("5.4 文献定位"),
    para("施展. 附加轮廓矢量的SAR建筑物精细地理编码[本科毕业设计初稿]. 同济大学测绘与地理信息学院, 2026. 本说明依据本地PDF全文逐页核对，重点对应正文第2.3、2.4、3.3—3.7、5.7和6.2节。"),
  );

  const file = path.join(OUT_PAPER, "施展论文借鉴作用与建筑估高方法辨析.docx");
  fs.writeFileSync(file, await Packer.toBuffer(docOptions("施展论文借鉴作用与建筑估高方法辨析", c)));
  return file;
}

(async () => {
  const combined = await buildCombined();
  const paper = await buildPaper();
  console.log(combined);
  console.log(paper);
})();
