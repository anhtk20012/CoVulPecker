import * as vscode from 'vscode';
import fetch from "node-fetch";

interface Prediction {
    range: string;
    pred: string;
    lines: number[];
    average : number;
}

export function extractMethodsFromFile(editor: vscode.TextEditor) {
    const document = editor.document;
    const code: string = document.getText();
    return code;
}

export async function prediction(requestBody: { lines: number[]; code: string; }) {
    const response = await fetch(
        'http://127.0.0.1:5000/predict/section',
        {
            method: 'POST',
            body: JSON.stringify(requestBody),
            headers: { 'Content-Type': 'application/json' }
        }
    );
    const predictions = await response.json();
    return predictions;
}

export function highlightcolor(predictions: any) {
    predictions.forEach((element: Prediction) => {
        const editor = vscode.window.activeTextEditor;
        const highlightDecorationType = vscode.window.createTextEditorDecorationType({
            backgroundColor: 'rgba(255, 255, 0, 0.3)'
        });
        const ranges = element.lines.map(lineNumber => {
            try {
                return editor?.document.lineAt(lineNumber - 1).range;
            } catch (error) {
                return null;
            }
        })
        .filter((range): range is vscode.Range => range !== null);;
        if (ranges.length > 0){
            editor?.setDecorations(highlightDecorationType, ranges);
        }
        else{
            editor?.setDecorations(highlightDecorationType, []);
        }
    });
}

