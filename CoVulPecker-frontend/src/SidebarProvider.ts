import * as vscode from "vscode";
import { extractMethodsFromFile, prediction, highlightcolor } from "./action/utils";

export class SidebarProvider implements vscode.WebviewViewProvider {
    _view?: vscode.WebviewView;
    _doc?: vscode.TextDocument;

    constructor(private readonly _extensionUri: vscode.Uri) { }

    public resolveWebviewView(webviewView: vscode.WebviewView) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };
        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
        webviewView.webview.onDidReceiveMessage(async (data) => {
            console.log('Message received in extension:', data);
            const editor = vscode.window.activeTextEditor;
        
            if (!editor) {
                webviewView.webview.postMessage({ command: 'onError', error: "Invalid text editor. Please open your file for scanning." });
                return;
            }

            const code_edit = editor?.document.languageId.toUpperCase();
            switch (code_edit){
                case "C":
                    break;
                case "CPP":
                    break;
                default:
                    webviewView.webview.postMessage({ command: 'onError', error: "Invalid language. Please ensure that the file contains C/C++ source code." });
                    return;
            }

            if (editor?.document.fileName.startsWith('Untitled')){
                webviewView.webview.postMessage({ command: 'onError', error: "Invalid file name. Please ensure that the file contains C/C++ source code." });
                return;
            }

            const filename = (editor.document.fileName).split('\\').pop();

            switch (data.type) {
                case "onCodeSelection": {
                    let start = new Date().getTime();
                    const selection = editor.selection;

                    const lineStart = selection.start.line;
                    const lineEnd = selection.end.line;

                    if (lineEnd - lineStart < 1) {
                        webviewView.webview.postMessage({ command: 'onError', error: "Please, select a bigger code section." });
                        return;
                    }

                    const code = editor.document.getText(selection);

                    let requestBody = {
                        lines: [lineStart, lineEnd],
                        code: code
                    };
                    
                    // (async () => {
                        try {
                            const predictions = await prediction(requestBody);
                            let end = new Date().getTime();
                            let time = end - start;
                            webviewView.webview.postMessage({ command: 'Prediction', filename: filename, predictions: predictions, time: time});
                            highlightcolor(predictions);
                            return;
                        } catch (error) {
                            webviewView.webview.postMessage({ command: 'onError', error: "Server error. Please review your access settings and configurations are correct." });
                            return;
                        }
                    // })();
                }
                case "onCompleteFile": {
                    let start = new Date().getTime();
                    const lineStart = 0;
                    const lineEnd = editor.document.lineCount - 1;
                    let requestBody = {
                        lines: [lineStart, lineEnd],
                        code: editor.document.getText()
                    };
                    // (async () => {
                        try {
                            const predictions = await prediction(requestBody);
                            let end = new Date().getTime();
                            let time = end - start;
                            webviewView.webview.postMessage({ command: 'Prediction', filename: filename, predictions: predictions, time: time});
                            
                            return;
                        } catch (error) {
                            webviewView.webview.postMessage({ command: 'onError', error: "Server error. Please review your access settings and configurations are correct." });
                            return;
                        }
                    // })();

                } 
            }
        });
    }

    public revive(panel: vscode.WebviewView) {
        this._view = panel;
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const styleResetUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, "media", "reset.css")
        );

        const styleVSCodeUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, "media", "vscode.css")
        );

        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, "out", "compiled/sidebar.js")
        );

        const styleMainUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, "out", "compiled/extra.css")
        );
        
        const nonce = getNonce();
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="Content-Security-Policy" content="img-src https: data:; style-src 'unsafe-inline' ${webview.cspSource}; script-src 'nonce-${nonce}';">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleResetUri}" rel="stylesheet">
            <link href="${styleVSCodeUri}" rel="stylesheet">
            <link href="${styleMainUri}" rel="stylesheet">
            <script nonce="${nonce}">
                    const vscode = acquireVsCodeApi();
                </script>
            <title>My Sidebar</title>
        </head>
        <body>
            <script nonce="${nonce}" src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}
function getNonce() {
    let text = "";
    const possible =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}