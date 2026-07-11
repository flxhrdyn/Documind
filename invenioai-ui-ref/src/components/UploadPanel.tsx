/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef } from 'react';
import { Upload, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { DocumentFile } from '../types';

interface UploadPanelProps {
  onDocumentAdded: (doc: DocumentFile) => void;
}

interface UploadJob {
  id: string;
  filename: string;
  size: string;
  status: 'pending' | 'parsing' | 'indexing' | 'succeeded' | 'failed';
  progress: number;
  error?: string;
}

export default function UploadPanel({ onDocumentAdded }: UploadPanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [activeJobs, setActiveJobs] = useState<UploadJob[]>([]);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setError(null);
    const files = Array.from(e.dataTransfer.files) as File[];
    processFiles(files);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    if (e.target.files) {
      const files = Array.from(e.target.files) as File[];
      processFiles(files);
    }
  };

  const processFiles = (files: File[]) => {
    const pdfs = files.filter(f => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'));
    
    if (pdfs.length === 0) {
      setError('Only PDF files are supported.');
      return;
    }

    pdfs.forEach(file => {
      // Validate file size (Max 15MB for demo sandbox)
      const maxSizeBytes = 15 * 1024 * 1024;
      if (file.size > maxSizeBytes) {
        setError(`File "${file.name}" exceeds the 15MB size limit.`);
        return;
      }

      startIndexingSimulation(file);
    });
  };

  const startIndexingSimulation = (file: File) => {
    const jobId = 'job-' + Date.now() + '-' + Math.floor(Math.random() * 1000);
    const newJob: UploadJob = {
      id: jobId,
      filename: file.name,
      size: formatBytes(file.size),
      status: 'pending',
      progress: 10,
    };

    setActiveJobs(prev => [newJob, ...prev]);

    // Stage 1: Pending (0 - 1.5s)
    setTimeout(() => {
      updateJobStatus(jobId, { status: 'parsing', progress: 35 });

      // Stage 2: Parsing (1.5s - 3.5s)
      setTimeout(() => {
        updateJobStatus(jobId, { status: 'indexing', progress: 70 });

        // Stage 3: Indexing (3.5s - 6s)
        setTimeout(() => {
          // Success! Create custom parsed chunks
          const finalDoc: DocumentFile = {
            id: 'doc-' + Date.now(),
            name: file.name,
            size: formatBytes(file.size),
            uploadedAt: new Date().toISOString().replace('T', ' ').substring(0, 16),
            chunksCount: 3,
            status: 'succeeded',
            chunks: [
              {
                id: `chunk-${jobId}-1`,
                text: `Document "${file.name}" successfully indexed. This section details the executive summary covering key objectives discussed on the primary page of the document, as well as analytical methodologies.`,
                pageNumber: 1
              },
              {
                id: `chunk-${jobId}-2`,
                text: `Specific empirical analysis establishes a strong correlation between overall operational throughput and dual-layer semantic cache deployment. Core search latency dropped exponentially after 10 contiguous query rounds.`,
                pageNumber: 2
              },
              {
                id: `chunk-${jobId}-3`,
                text: `Conclusions and forward-looking guidelines recommend a standardized deployment of the hybrid dense-sparse retriever to balance query workloads efficiently across secondary vector edge replicas.`,
                pageNumber: 3
              }
            ]
          };

          updateJobStatus(jobId, { status: 'succeeded', progress: 100 });
          onDocumentAdded(finalDoc);

          // Clear completed job after a delay
          setTimeout(() => {
            setActiveJobs(prev => prev.filter(j => j.id !== jobId));
          }, 4000);

        }, 2500);
      }, 2000);
    }, 1500);
  };

  const updateJobStatus = (jobId: string, updates: Partial<UploadJob>) => {
    setActiveJobs(prev =>
      prev.map(job => (job.id === jobId ? { ...job, ...updates } : job))
    );
  };

  const getStatusMessage = (status: UploadJob['status']) => {
    switch (status) {
      case 'pending': return 'Preparing file...';
      case 'parsing': return 'Parsing PDF content...';
      case 'indexing': return 'Analyzing text...';
      case 'succeeded': return 'Document added!';
      case 'failed': return 'Failed to add document.';
      default: return '';
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200 font-display">Upload Document</h3>
        <span className="text-xs text-gray-400 dark:text-gray-500 font-sans">Max 15MB</span>
      </div>

      {/* Drag & Drop Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`group relative flex flex-col items-center justify-center border border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300 ${
          isDragging
            ? 'border-teal-500 bg-teal-50/20 dark:border-teal-400 dark:bg-teal-950/20'
            : 'border-gray-200 hover:border-gray-300 dark:border-gray-800 dark:hover:border-gray-700 bg-white/40 dark:bg-gray-900/40 hover:bg-white dark:hover:bg-gray-900'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={handleFileSelect}
        />

        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-xl mb-4 group-hover:scale-105 transition-transform duration-200 shadow-sm border border-gray-100 dark:border-gray-800">
          <Upload className="w-5 h-5 text-gray-400 dark:text-gray-300 group-hover:text-teal-500 transition-colors duration-200" />
        </div>

        <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Drag & drop PDF here, or <span className="text-teal-600 dark:text-teal-500 hover:underline">browse file</span>
        </p>
        <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
          Supports standard PDF files
        </p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="flex items-start space-x-2 p-3 rounded-lg bg-rose-50 dark:bg-rose-950/20 text-rose-800 dark:text-rose-300 border border-rose-100 dark:border-rose-950">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <span className="text-xs font-medium">{error}</span>
        </div>
      )}

      {/* Active Upload/Indexing Jobs */}
      {activeJobs.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500">
            Uploading
          </h4>
          <div className="space-y-2">
            {activeJobs.map(job => (
              <div
                key={job.id}
                className="p-3 rounded-xl border border-gray-200/60 dark:border-gray-800/60 bg-white dark:bg-gray-900 shadow-sm"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2 min-w-0">
                    <FileText className="w-4 h-4 text-gray-400 shrink-0" />
                    <span className="text-xs font-medium text-gray-800 dark:text-gray-200 truncate max-w-[160px]">
                      {job.filename}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-gray-400 dark:text-gray-500">
                    {job.size}
                  </span>
                </div>

                {/* Progress bar */}
                <div className="w-full bg-gray-100 dark:bg-gray-800 h-1.5 rounded-full overflow-hidden mb-2">
                  <div
                    className={`h-full transition-all duration-500 rounded-full ${
                      job.status === 'succeeded'
                        ? 'bg-emerald-500'
                        : job.status === 'failed'
                        ? 'bg-rose-500'
                        : 'bg-teal-500'
                    }`}
                    style={{ width: `${job.progress}%` }}
                  />
                </div>

                <div className="flex items-center justify-between text-[10px] text-gray-500 dark:text-gray-400">
                  <div className="flex items-center space-x-1.5 font-mono">
                    {job.status !== 'succeeded' && job.status !== 'failed' && (
                      <Loader2 className="w-3 h-3 text-teal-500 animate-spin" />
                    )}
                    {job.status === 'succeeded' && (
                      <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                    )}
                    {job.status === 'failed' && (
                      <AlertCircle className="w-3 h-3 text-rose-500" />
                    )}
                    <span className="font-sans font-medium">{getStatusMessage(job.status)}</span>
                  </div>
                  <span className="font-mono">{job.progress}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
