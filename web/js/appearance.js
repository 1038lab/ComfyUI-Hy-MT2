import { app } from "/scripts/app.js";

const COLOR_THEMES = {
    hymt2: { nodeColor: "#28403f", nodeBgColor: "#28403f" },
};

const NODE_COLORS = {
    "HyMT_Translation_GGUF":     "hymt2",
};

function setNodeColors(node, theme) {
    if (!theme) return;
    if (theme.nodeColor) node.color = theme.nodeColor;
    if (theme.nodeBgColor) node.bgcolor = theme.nodeBgColor;
    if (theme.title_color) node.title_color = theme.title_color;
    if (theme.width) {
        node.size = node.size || [140, 80];
        node.size[0] = theme.width;
    }
}

function applyTheme(node) {
    if (!node) return;
    const nclass = node.comfyClass || node.type;
    if (NODE_COLORS.hasOwnProperty(nclass)) {
        const colorKey = NODE_COLORS[nclass];
        const theme = COLOR_THEMES[colorKey];
        setNodeColors(node, theme);
    }
}

const ext = {
    name: "HyMT.appearance",
    nodeCreated(node) {
        applyTheme(node);
    },
    loadedGraphNode(node) {
        applyTheme(node);
    }
};

app.registerExtension(ext);
