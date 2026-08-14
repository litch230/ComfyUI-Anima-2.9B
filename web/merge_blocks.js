import { app } from "/scripts/app.js";

const INSERTED_BLOCKS = new Set([2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36]);

function roundedRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
        ctx.roundRect(x, y, width, height, radius);
    } else {
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
    }
    ctx.closePath();
}

function lockBlockWidget(widget) {
    widget.value = 1.0;
    widget.disabled = true;
    widget.options = {
        ...widget.options,
        min: 1.0,
        max: 1.0,
        disabled: true,
    };

    widget.callback = () => {
        widget.value = 1.0;
    };
    widget.serializeValue = () => 1.0;
    widget.mouse = () => true;

    widget.draw = function (ctx, node, widgetWidth, widgetY, height) {
        const margin = 10;
        const rowY = widgetY + 1;
        const rowHeight = height - 2;

        ctx.save();
        roundedRect(ctx, margin, rowY, widgetWidth - margin * 2, rowHeight, rowHeight / 2);
        ctx.fillStyle = "#202020";
        ctx.fill();
        ctx.strokeStyle = "#303030";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = "#666666";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(this.name, margin + 10, rowY + rowHeight / 2);
        ctx.restore();
    };
}

app.registerExtension({
    name: "Anima2.9B.MergeBlocks.LockInsertedBlocks",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AnimaExpandedModelMergeBlocks") {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);

            for (const blockIndex of INSERTED_BLOCKS) {
                const widget = this.widgets?.find(
                    (candidate) => candidate.name === `blocks.${blockIndex}.`,
                );
                if (widget) {
                    lockBlockWidget(widget);
                }
            }

            return result;
        };
    },
});
