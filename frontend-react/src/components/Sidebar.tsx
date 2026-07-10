import UploadPanel from './UploadPanel';
import KnowledgeBaseList from './KnowledgeBaseList';

export default function Sidebar() {
  return (
    <div className="flex flex-col gap-6">
      <UploadPanel />
      <div className="border-t border-line pt-5">
        <KnowledgeBaseList />
      </div>
    </div>
  );
}
