import tempfile
import unittest
from pathlib import Path
from artifacts import freeze, restore, unchanged


class Artifacts(unittest.TestCase):
    def test_fresh_restore_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';target=root/'target';bundle=root/'bundle'
            for relative in ['pyproject.toml','uv.lock',
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
            (target/'uv.lock').write_text('changed')
            self.assertFalse(unchanged(target,bundle))
            (bundle/'content/.kinex/tool/solve_task/tool.py').write_text('tampered')
            with self.assertRaises(ValueError):restore(bundle,target,identity={'kinex':'revision'})


if __name__=='__main__':unittest.main()
