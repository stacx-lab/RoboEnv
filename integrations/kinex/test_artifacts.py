import tempfile
import unittest
from pathlib import Path
from artifacts import freeze, restore, unchanged


class Artifacts(unittest.TestCase):
    def test_freeze_requires_core_lock_and_excludes_virtualenv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bundle = root / 'source', root / 'bundle'
            tool = source / '.kinex/tool/solve_task'
            tool.mkdir(parents=True)
            (tool / 'tool.yaml').write_text('name: solve_task\n')
            for name in ('pyproject.toml', 'uv.lock'):
                (source / name).write_text('host environment\n')
            with self.assertRaisesRegex(ValueError, 'Python environment must be locked'):
                freeze(source, bundle, identity={}, tool='solve_task', arguments={})
            self.assertFalse(bundle.exists())
            core = source / '.kinex/python'
            (core / '.venv').mkdir(parents=True)
            (core / 'pyproject.toml').write_text('managed environment\n')
            (core / 'uv.lock').write_text('locked dependencies\n')
            (core / '.venv/private').write_text('do not freeze\n')
            record = freeze(source, bundle, identity={}, tool='solve_task', arguments={})
            self.assertIn('.kinex/python/uv.lock', record['files'])
            self.assertNotIn('.kinex/python/.venv/private', record['files'])
            self.assertFalse((bundle / 'content/.kinex/python/.venv').exists())

    def test_fresh_restore_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';target=root/'target';bundle=root/'bundle'
            for relative in ['.kinex/python/pyproject.toml','.kinex/python/uv.lock',
                             '.kinex/tool/solve_task/tool.yaml','.kinex/tool/solve_task/tool.py',
                             '.kinex/devices/arm/profile.yaml']:
                p=source/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixed\n')
            (source/'.kinex/devices/arm/connection.json').write_text('private episode')
            freeze(source,bundle,identity={'kinex':'revision'},tool='solve_task',arguments={})
            p=target/'.kinex/devices/arm/profile.yaml';p.parent.mkdir(parents=True);p.write_text('fixed\n')
            restore(bundle,target,identity={'kinex':'revision'})
            self.assertTrue(unchanged(target,bundle))
            self.assertFalse((target/'.kinex/devices/arm/connection.json').exists())
            with self.assertRaises(ValueError):
                restore(bundle,target,identity={'kinex':'changed dependency'})
            # New development may repair old code under revalidated infrastructure.
            # Final validation above remains strict about dependency identity.
            restore(bundle,target,identity={'kinex':'changed dependency'},development=True)
            (target/'.kinex/python/uv.lock').write_text('changed')
            self.assertFalse(unchanged(target,bundle))
            (bundle/'content/.kinex/tool/solve_task/tool.py').write_text('tampered')
            with self.assertRaises(ValueError):restore(bundle,target,identity={'kinex':'revision'})


if __name__=='__main__':unittest.main()
