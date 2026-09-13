import inspect
import unittest
from src.data.natural_support import natural_support_manifest


class NaturalSupportTest(unittest.TestCase):
    def fixture(self,n):
        return {str(i):dict(train=list(range(i*100,i*100+n)),test=[i*100+90]) for i in range(100)}

    def test_common_size_exclusions_and_determinism(self):
        clients=self.fixture(65);excluded={i*100 for i in range(100)}
        manifest=natural_support_manifest(clients,excluded)
        self.assertEqual(manifest['chosen_K'],64);self.assertEqual(manifest['secondary_K'],20)
        self.assertEqual(manifest,natural_support_manifest(clients,excluded))
        for cid,r in manifest['clients'].items():
            self.assertEqual(len(r['selected']),64)
            self.assertTrue(all(x['original_id'] not in excluded for x in r['selected']))
            self.assertTrue(all(clients[cid]['train'][x['train_index']]==x['original_id'] for x in r['selected']))
        self.assertEqual(list(inspect.signature(natural_support_manifest).parameters),['clients','calibration_ids'])

    def test_common_prefix_and_unavailable_size(self):
        clients=self.fixture(31);m=natural_support_manifest(clients,set())
        self.assertEqual(m['chosen_K'],20);self.assertIsNone(m['secondary_K'])
        clients['0']['train']=clients['0']['train'][:15]
        m=natural_support_manifest(clients,set());self.assertIsNone(m['chosen_K']);self.assertEqual(m['eligible_counts']['0'],15)


if __name__=='__main__':unittest.main()
