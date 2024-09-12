import * as vscode from 'vscode';

import { SidebarProvider } from './SidebarProvider';

export function activate(context: vscode.ExtensionContext) {
	const sidebarProvider = new SidebarProvider(context.extensionUri);
	context.subscriptions.push(
		vscode.commands.registerCommand('covulpecker.helloWorld', () => {
			vscode.window.showInformationMessage('Hello CoVulPecker!');
		})
	);
	context.subscriptions.push(
		vscode.window.registerWebviewViewProvider(
			"covulpecker.sidebar",
			sidebarProvider
		)
	);
}

export function deactivate() {}
