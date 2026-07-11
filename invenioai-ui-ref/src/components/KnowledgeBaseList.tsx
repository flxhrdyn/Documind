/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { FileText, Trash2, Database, AlertTriangle, ShieldAlert } from 'lucide-react';
import { DocumentFile } from '../types';

interface KnowledgeBaseListProps {
  documents: DocumentFile[];
  onDeleteDocument: (id: string) => void;
  onClearAllDocuments: () => void;
}

export default function KnowledgeBaseList({
  documents,
  onDeleteDocument,
  onClearAllDocuments
}: KnowledgeBaseListProps) {
  const [docToDelete, setDocToDelete] = useState<DocumentFile | null>(null);
  const [showClearAllModal, setShowClearAllModal] = useState(false);

  const confirmDeleteSingle = (doc: DocumentFile) => {
    setDocToDelete(doc);
  };

  const confirmClearAll = () => {
    setShowClearAllModal(true);
  };

  const handleConfirmSingleDelete = () => {
    if (docToDelete) {
      onDeleteDocument(docToDelete.id);
      setDocToDelete(null);
    }
  };

  const handleConfirmClearAll = () => {
    onClearAllDocuments();
    setShowClearAllModal(false);
  };

  return (
    <div className="space-y-4">
      {/* List Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Database className="w-4 h-4 text-teal-600 dark:text-teal-400" />
          <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200 font-display">Knowledge Base</h3>
        </div>
        {documents.length > 0 && (
          <button
            onClick={confirmClearAll}
            className="text-xs font-medium text-rose-500 hover:text-rose-600 dark:text-rose-400 dark:hover:text-rose-300 transition-colors duration-150 cursor-pointer"
          >
            Clear All
          </button>
        )}
      </div>

      {/* Empty State for documents */}
      {documents.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-10 px-4 rounded-xl border border-gray-100 dark:border-gray-900 bg-white/20 dark:bg-gray-900/10 text-center">
          <Database className="w-6 h-6 text-gray-300 dark:text-gray-700 mb-2" />
          <p className="text-xs font-medium text-gray-600 dark:text-gray-400">No documents uploaded yet</p>
          <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-0.5">Upload a PDF file above to start questioning</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
          {documents.map(doc => (
            <div
              key={doc.id}
              className="group flex items-center justify-between p-2.5 rounded-xl border border-gray-100 dark:border-gray-900 bg-white dark:bg-gray-900/60 hover:bg-gray-50/80 dark:hover:bg-gray-900 transition-all duration-200 shadow-sm"
            >
              <div className="flex items-center space-x-2.5 min-w-0">
                <div className="p-1.5 bg-teal-50 dark:bg-teal-950/20 text-teal-600 dark:text-teal-400 rounded-lg shrink-0">
                  <FileText className="w-3.5 h-3.5" />
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-medium text-gray-700 dark:text-gray-200 truncate pr-2 max-w-[170px]" title={doc.name}>
                    {doc.name}
                  </p>
                  <div className="flex items-center space-x-2 text-[10px] text-gray-400 dark:text-gray-500 mt-0.5 font-mono">
                    <span>{doc.size}</span>
                    <span>•</span>
                    <span>{doc.chunksCount} passages</span>
                  </div>
                </div>
              </div>

              <button
                onClick={() => confirmDeleteSingle(doc)}
                className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1.5 hover:bg-rose-50 dark:hover:bg-rose-950/30 text-gray-400 hover:text-rose-500 dark:text-gray-600 dark:hover:text-rose-400 rounded-lg transition-all duration-150 cursor-pointer"
                aria-label={`Delete document ${doc.name}`}
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* CUSTOM DESTRUCTIVE MODAL: Single Delete Confirmation */}
      {docToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-950/60 backdrop-blur-xs">
          <div className="w-full max-w-sm bg-white dark:bg-gray-950 border border-gray-200 dark:border-gray-900 rounded-2xl p-6 shadow-xl animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center space-x-3 text-teal-600 dark:text-teal-400 mb-4">
              <div className="p-2 bg-teal-50 dark:bg-teal-950/30 rounded-xl">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100">Delete Document?</h4>
            </div>
            
            <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed mb-4 font-sans">
              You are about to delete document <strong className="text-gray-800 dark:text-gray-200">"{docToDelete.name}"</strong>. It will be permanently removed from the list.
            </p>

            <div className="flex items-center justify-end space-x-2.5">
              <button
                onClick={() => setDocToDelete(null)}
                className="px-3.5 py-1.5 text-xs font-medium text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-xl transition-all duration-150 cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmSingleDelete}
                className="px-3.5 py-1.5 text-xs font-medium text-white bg-rose-600 hover:bg-rose-700 dark:bg-rose-700 dark:hover:bg-rose-600 rounded-xl transition-all duration-150 cursor-pointer shadow-sm"
              >
                Delete Permanently
              </button>
            </div>
          </div>
        </div>
      )}

      {/* CUSTOM DESTRUCTIVE MODAL: Clear All Confirmation */}
      {showClearAllModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-950/60 backdrop-blur-xs">
          <div className="w-full max-w-sm bg-white dark:bg-gray-950 border border-gray-200 dark:border-gray-900 rounded-2xl p-6 shadow-xl animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center space-x-3 text-rose-600 dark:text-rose-500 mb-4">
              <div className="p-2 bg-rose-50 dark:bg-rose-950/30 rounded-xl">
                <ShieldAlert className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100">Clear Entire Index?</h4>
            </div>

            <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed mb-4 font-sans">
              This action will delete all <strong className="text-gray-800 dark:text-gray-200">({documents.length}) documents</strong>. Future answers will not have access to these files.
            </p>

            <div className="flex items-center justify-end space-x-2.5">
              <button
                onClick={() => setShowClearAllModal(false)}
                className="px-3.5 py-1.5 text-xs font-medium text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-xl transition-all duration-150 cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmClearAll}
                className="px-3.5 py-1.5 text-xs font-medium text-white bg-rose-600 hover:bg-rose-700 dark:bg-rose-700 dark:hover:bg-rose-600 rounded-xl transition-all duration-150 cursor-pointer shadow-sm"
              >
                Delete All Documents
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
