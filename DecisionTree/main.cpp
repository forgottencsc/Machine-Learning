#include <algorithm>
#include <functional>
#include <vector>
#include <queue>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstring>

#include <numeric>
#include <random>
#include <chrono>

#include <ctime>

#include <Eigen/Dense>

#define BUILD_INFO 0
#define PRUNE_INFO 1
namespace ml {

    using std::pair;
	using std::vector;
	using std::string;
	using std::priority_queue;
	using std::function;

	using std::max;
	using std::max_element;
	using std::accumulate;
	using std::copy_n;
	using std::distance;
	using std::fill;
	using std::partition;
	using std::sort;
	using std::unique;

	using std::mt19937_64;
	using std::shuffle;

	using std::runtime_error;
	using std::cin;
	using std::cout;
	using std::endl;

	//  first是熟悉，second是标签
	typedef pair<vector<float>, int> data;

	//  用vector存储数据集
    typedef vector<data> dataset_t;

    //  分类器接收一条数据，并返回一个整数
    typedef function<int(const data& d)> classifier;

	class decision_tree {
	public:

        //  放在节点上的分类器，一条数据的第i个属性值是否小于v
        struct attr_clf {
            float v; int i;
            int operator()(const data& d) const {
                return d.first[i] < v;
            }
        };

        //  从每条数据中提取标签的Functor
        struct lbl_clf {
            int operator()(const data& d) const {
                return d.second;
            }
        };

        //  决策树节点，若is_leaf为真则res代表当前节点的分类结果
		struct node {
			node() : is_leaf(false) { c[0] = c[1] = 0; }
			node* c[2];
			classifier f;
			int res;
			bool is_leaf;
		};

		int attr_cnt;	    //	属性总数
		classifier clf0;	//  主分类器，用于提取标签
		node* root;         //  决策树根节点

		//  销毁以p为根的子树
		void destroy(node*& p) {
			if (!p) return;
			else if (!p->is_leaf) {
                destroy(p->c[0]);
                destroy(p->c[1]);
			}
			delete p;
			p = 0;
		}

		//  统计代器区间[first,last)中的数据类型，将结果存入c中
		template<class BidIt>
		void count(int* c, BidIt first, BidIt last) {
            fill(c, c + 10, 0);
			for (auto it = first; it != last; ++it)
				c[clf0(*it)]++;
		}

        //  统计信息熵，s为总数
        double cal_ent(int* c, int s) {
            double res = 0;
            for (int i = 0; i != 10; ++i) {
                if (!c[i]) continue;    //  不存在第i类样本，跳过
                double p = (double)c[i] / s;
                res -= p * log(p);
            }
            return res;
        }

        //  统计信息增益
        double cal_gain(int* c, int* lc, int* rc) {
            int s = 0, ls = 0, rs = 0;
            //  统计被分类至左子树，右子树中的数据条数
            for (int i = 0; i != 10; ++i) {
                s += c[i];
                ls += lc[i];
                rs += rc[i];
            }
            //  计算信息增益
            double e = cal_ent(c, s),
                   e1 = cal_ent(lc, ls),
                   e0 = cal_ent(rc, rs);
            if (!ls) e1 = 0;
            if (!rs) e0 = 0;
            return e - (ls * e1 + rs * e0) / (ls + rs);
        }

        //  老版本的信息增益计算，复杂度过大
//		template<class BidIt>
//		double cal_gain(const classifier& f, BidIt first, BidIt last) {
//			auto it = partition(first, last, f);
//			size_t D1 = distance(first, it), D0 = distance(it, last);
//			double e = cal_ent(first, last),
//				e0 = cal_ent(it, last),
//				e1 = cal_ent(first, it);
//			if (!D0) e0 = 0;
//			if (!D1) e1 = 0;
//			return e - (D1 * e1 + D0 * e0) / (D1 + D0);
//		}

        //  选取标签众数作为分类结果
		static inline int cal_res(int* c) {
		    return max_element(c, c + 10) - c; //c[0] <= c[1] ? 1 : 0;
        }

        //  计算选取众数为分类结果时的分类精准度
		static inline double cal_acc(int* c) {
		    int p = cal_res(c);
            if (c[p] == 0) return 1.0;
            else return (double)(c[p]) / accumulate(c, c + 10, 0);
		}

		//  判断是否仅剩下一类样本
		static inline bool single_class(int* c) {
            return *max_element(c, c + 10) == accumulate(c, c + 10, 0);
        }

        //  在迭代器区间[first,last)中寻找信息增益最大的分类器，将结果存入res
        //  老版本在统计每个分割点时都需要遍历整个迭代器区间，在MNIST数据集上运行速度极慢
        //  老版本时间复杂度为O(attr_cnt * dist(first,last) ^ 2)
        //  新版本在统计每个分割点的信息时能O(1)转移到下个分割点，性能比老版本有极大的提升
        //  新版本时间复杂度为O(attr_cnt * dist(first,last) * log(dist(first,last)))
		template<class BidIt>
		bool max_gain_classifier(classifier& res, BidIt first, BidIt last) {
			double max_gain = 0;    //  最大增益
			bool flag = 0;	        //	标记为真时说明存在至少一个候选分类器
			int c[10]; count(c, first, last);   //  统计区间内数据标签个数
			//  依次考虑所有属性
			for (int i = 0; i != attr_cnt; ++i) {
                //  因为是连续型属性，所以按第i个属性值大小进行排序
                sort(first, last, [i](const ml::data& d1, const ml::data& d2) {
                     return d1.first[i] < d2.first[i];
                });

                //  初始状态为左子树为空，全部分类至右子树。
                int lc[10] = {0}, rc[10] = {0}; copy_n(c, 10, rc);
                for (auto it = first; it != prev(last); ++it) {
                    //  *it与*next(it)在第i个属性上的值相等，无法分割
                    if (it->first[i] == next(it)->first[i]) continue;
                    //  选取分割点为*it与*next(it)在第i个属性上的值的平均值
                    float v = (it->first[i] + (next(it)->first[i])) / 2;
                    //  构造分类器
                    classifier f = attr_clf({ v, i });
                    //  *it被从右子树移出并加入左子树
                    lc[clf0(*it)]++;
                    rc[clf0(*it)]--;
                    //  计算当前信息增益
                    double gain = cal_gain(c, lc, rc);
                    if (gain > max_gain) {
                        max_gain = gain;
                        res = f;
                        flag = 1;
                    }
                }

                //  老版本的信息增益计算，复杂度过大
//				for (int j = 0; j != (int)s.size() - 1; ++j) {
//					float v = (s[j] + s[j + 1]) / 2.;
//					classifier f = attr_clf{ v, i };
//					double gain = cal_gain(f, first, last);
//					if (gain > max_gain) {
//						max_gain = gain;
//						res = f;
//						flag = 1;
//					}
//				}
			}
			return flag;
		}

		//  对p建立新节点，并对迭代器区间[first,last)中的元素进行划分
		template<class BidIt>
		void build(node*& p, BidIt first, BidIt last) {
		    #if BUILD_INFO
		    cout << "Build! " << distance(first, last) << endl;
		    #endif
			if (first == last) throw runtime_error("Empty range!");
            //  统计数据标签个数
			int c[10]; count(c, first, last);
			p = new node();
			classifier f;
			if (single_class(c)) {
                //  标签仅剩一种，设为叶节点并结束
                #if BUILD_INFO
                cout << "Single class!" << endl;
                #endif
				p->res = clf0(*first);
				p->is_leaf = true;
			}
			else if (!max_gain_classifier(f, first, last)) {
			    //  无法选取分类器，设为叶节点并将标签众数设为分类结果
                #if BUILD_INFO
                cout << "Classifier selection failed!" << endl;
                #endif
				p->res = cal_res(c);
				p->is_leaf = true;
			}
			else {
			    //  成功选取分类器，划分后递归建树
				p->f = f;
				auto it = partition(first, last, f);
				build(p->c[1], first, it);
				build(p->c[0], it, last);
			}
		}

		template<class BidIt>
		void pre_pruning_build(node*& p,
			BidIt first, BidIt last, BidIt pfirst, BidIt plast) {
			if (first == last) throw runtime_error("Empty range!");
			int c[10]; count(c, first, last);
			int pc[10]; count(pc, pfirst, plast);

			p = new node;
			classifier f;
			if (single_class(c)) {
				p->res = clf0(*first);
				p->is_leaf = true;
			}
			else if (!max_gain_classifier(f, first, last)) {
				p->res = cal_res(c);
				p->is_leaf = true;
			}
			else {
				auto it = partition(first, last, f);
				auto pit = partition(pfirst, plast, f);
				int plc[10]; count(plc, pfirst, pit);
				int prc[10]; count(prc, pit, plast);
				int lc[10]; count(lc, first, it);
				int rc[10]; count(rc, it, last);
				int pres = cal_res(c);
				double acc_prune = (double)pc[pres] / distance(pfirst, plast);
				double acc_not_prune = (double)(plc[cal_res(lc)] + prc[cal_res(rc)]) / distance(pfirst, plast);
				if (acc_prune > acc_not_prune) {
#if PRUNE_INFO
					cout << "Pre-pruned! " << acc_not_prune << ' ' << acc_prune << endl;
#endif
					p->res = pres;
					p->is_leaf = true;
				}
				else {
					p->f = f;
					pre_pruning_build(p->c[1], first, it, pfirst, pit);
					pre_pruning_build(p->c[0], it, last, pit, plast);
				}
			}
		}

		template<class BidIt>
		double post_pruning_build(node*& p,
			BidIt first, BidIt last, BidIt pfirst, BidIt plast) {
			if (first == last) throw runtime_error("Empty range!");
			int c[10]; count(c, first, last);
			int pc[10]; count(pc, pfirst, plast);
			p = new node;
			classifier f;
			if (single_class(c)) {
				p->res = clf0(*first);
				p->is_leaf = true;
				int sp = distance(pfirst, plast);
				if (!sp) return 1.0;
				else return (double)pc[p->res] / sp;	//	acc
			}
			else if (!max_gain_classifier(f, first, last)) {
				p->res = cal_res(c);
				p->is_leaf = true;
				int sp = distance(pfirst, plast);
				if (!sp) return 1;
				else return (double)pc[p->res] / sp;	//	acc
			}
			else {
				p->f = f;
				auto it = partition(first, last, f);
				auto pit = partition(pfirst, plast, f);
				double acc_left = post_pruning_build(p->c[1], first, it, pfirst, pit);
				double acc_right = post_pruning_build(p->c[0], it, last, pit, plast);
				int num_left = distance(pfirst, pit);
				int num_right = distance(pit, plast);
				double acc_not_prune = (acc_left * num_left + acc_right * num_right) / (num_left + num_right);
				double acc_prune = (double)pc[cal_res(c)] / distance(pfirst, plast);
				if (acc_prune > acc_not_prune) {
#if PRUNE_INFO
					cout << "Post-pruned! " << acc_not_prune << ' ' << acc_prune << endl;
#endif
					destroy(p->c[1]); p->c[1] = 0;
					destroy(p->c[0]); p->c[0] = 0;
					p->is_leaf = true;
					p->res = cal_res(c);
					return acc_prune;
				}
				else
					return acc_not_prune;
			}
		}

		template<class BidIt>
		void build(BidIt first, BidIt last) {
		    destroy(root);
		    attr_cnt = first->first.size();
			build(root, first, last);
		}

		void build(dataset_t& v) {
			build(v.begin(), v.end());
		}

		template<class BidIt>
		void pre_pruning_build(BidIt first, BidIt last, BidIt pfirst, BidIt plast) {
		    destroy(root);
		    attr_cnt = first->first.size();
			pre_pruning_build(root, first, last, pfirst, plast);
		}

		void pre_pruning_build(dataset_t& v, dataset_t& ev) {
			pre_pruning_build(v.begin(), v.end(), ev.begin(), ev.end());
		}

		template<class BidIt>
		void post_pruning_build(BidIt first, BidIt last, BidIt pfirst, BidIt plast) {
		    destroy(root);
		    attr_cnt = first->first.size();
			post_pruning_build(root, first, last, pfirst, plast);
		}

		void post_pruning_build(dataset_t& v, dataset_t& ev) {
			post_pruning_build(v.begin(), v.end(), ev.begin(), ev.end());
		}

		decision_tree(classifier f0 = lbl_clf{}) : clf0(f0), root(0) {}

		~decision_tree() { destroy(root); root = 0; }

		//  统计树高
		int height(node* p) const {
			if (p->is_leaf) return 1;
			else return max(height(p->c[0]), height(p->c[1])) + 1;
		}

		int max_height() const {
			return height(root);
		}

        //  对数据进行预测
		int predict(const data& d) const {
			node* p = root;
			if (!p) throw std::runtime_error("Decision tree not built!");
			//  按每个节点的分类结果行进至叶节点
			while (!p->is_leaf)
				p = p->c[p->f(d)];
			return p->res;
		}

	};

	//  主成分分析部分
    using Eigen::Matrix;
	using Eigen::Dynamic;
    using Eigen::EigenSolver;
    typedef Matrix<double, Dynamic, Dynamic> Mat;
    typedef Matrix<double, Dynamic, 1> Vec;

    class PCA {
    public:
        //  W为投影矩阵
        Mat W, Wt;
        //  xc为训练集中心
        Vec xc;
        //  k为降维的目标维数
        int k;

        //  用训练集计算投影矩阵
        void build(const dataset_t& trainset, int k_) {
            int m = trainset[0].first.size();
            int n = trainset.size();
            k = k_;

            //  将数据导入矩阵
            Mat X(m, n);
            for (int i = 0; i != n; ++i)
                for (int j = 0; j != m; ++j)
                    X(j, i) = trainset[i].first[j];

            //  中心化
            xc = X * (Vec::Ones(n) / n);
            for (int i = 0; i != n; ++i)
                for (int j = 0; j != m; ++j)
                    X(j, i) -= xc(j);

            //  计算X*X^T
            Mat tmp = X * X.transpose();
            #if BUILD_INFO
            cout << "Computing Eigenvectors: ";
            #endif

            //  计算特征值与特征向量
            EigenSolver<Mat> ES(tmp, true);
            auto eigenvalues = ES.eigenvalues();
            auto eigenvectors = ES.eigenvectors();

            #if BUILD_INFO
            cout << "Complete!" << endl;
            #endif

            //for (int i = 0; i != m; ++i) cout << eigenvalues(i) << ' ';
            //cout << string(60, '$') << endl;

            //  用优先队列统计特征值最大的k个特征向量，特征值小的先出队
            priority_queue<int, vector<int>, function<bool(int, int)>>
            pq([&eigenvalues](int i1, int i2) { return eigenvalues(i1).real() > eigenvalues(i2).real(); });
            for (int i = 0; i != m; ++i) {
                pq.push(i);
                if (pq.size() > k) pq.pop();
            }

            //  特征向量的顺序无关紧要
            //  计算投影矩阵W
            W = Mat(m, k);
            for (int i = 0; i != k; ++i) {
                int t = pq.top(); pq.pop();
                //cout << (long long)eigenvalues(t).real() << " \n"[i == k - 1];
                for (int j = 0; j != m; ++j)
                    W(j, i) = eigenvectors.col(t)(j).real();
            }
            Wt = W.transpose();
        }

        //  将高维数据投影至低维
        dataset_t transform(const dataset_t& dataset) {
            int m = dataset[0].first.size(), n = dataset.size();
            dataset_t res(n, data(vector<float>(k, 0), 0));
            for (int i = 0; i != dataset.size(); ++i) {
                //  将数据导入向量
                Vec x(m), xv;
                for (int j = 0; j != m; ++j)
                    x(j) = dataset[i].first[j];
                //  投影
                xv = Wt * x;
                //  将投影后的数据存入结果
                for (int j = 0; j != k; ++j)
                    res[i].first[j] = xv(j);
                res[i].second = dataset[i].second;
            }
            return res;
        }

    };

    //  分割数据集，分隔点在split_point内
    vector<dataset_t> split_dataset(dataset_t& dataset, vector<double> split_point) {
        //  将数据集打乱
        mt19937_64 mt(time(0));
        shuffle(dataset.begin(), dataset.end(), mt);
        //  用vector存储多个数据集
        vector<dataset_t> res;
        int n = dataset.size();
        double p = 0;
        for (int i = 0; i != split_point.size(); ++i) {
            //  将[l,r)间的数据分入一个数据集
            int l = n * p, r = n * (p + split_point[i]);
            res.push_back(dataset_t(dataset.begin() + l, dataset.begin() + r));
            //  计算已经分割的比例
            p += split_point[i];
        }
        //  分入最后一个
        int l = n * p, r = n;
        res.push_back(dataset_t(dataset.begin() + l, dataset.begin() + r));
        return res;
    }

    //  最终的训练模型
    class model {
    public:
        decision_tree tree;
        PCA pca;
        int k;

        model(int k_) : k(k_) {}

        //  flg=0为不剪枝，1为预剪枝，2为后剪枝，1-p为验证集比例（不剪枝时该参数无效）
        void train(dataset_t& trainset, int flg = 0, double p = 0.6) {
            pca.build(trainset, k);
            dataset_t trainset_ = pca.transform(trainset);
            if (flg == 0)
                tree.build(trainset_.begin(), trainset_.end());
            else {
                int n = trainset_.size();
                auto it = trainset_.begin() + n * p;
                if (flg == 1)
                    tree.pre_pruning_build(trainset_.begin(), it, it, trainset_.end());
                else
                    tree.post_pruning_build(trainset_.begin(), it, it, trainset_.end());
            }
        }

        //  预测测试集的结果
        vector<int> predict(const dataset_t& testset) {
            dataset_t testset_ = pca.transform(testset);
            vector<int> res;
            for (int i = 0; i != testset_.size(); ++i)
                res.push_back(tree.predict(testset_[i]));
            return res;
        }

        //  评估模型的精度，p2为训练集所占比例（若flg不为0则包括验证集）
        double evaluate(dataset_t& dataset, int flg, double p = 0.6, double p2 = 0.8) {
            //  分割数据集
            vector<dataset_t> datasets = split_dataset(dataset, { p2 });
            dataset_t& trainset = datasets[0];
            dataset_t& testset = datasets[1];
            //  进行训练
            train(trainset, flg, p);
            //  预测结果
            vector<int> result = predict(testset);
            //  统计精准度
            int cnt = 0, tot = testset.size();
            for (int i = 0; i != tot; ++i)
                if (result[i] == testset[i].second)
                    cnt++;
            return (double)cnt / tot;
        }

        //  获取决策树的树高
        int get_height() {
            return tree.max_height();
        }
    };

}

using namespace std;

//  读入一项
string read(istream& is) {
	string res;
	char ch = is.get();
	while (ch != ',' && ch != '\n') {
		res += ch;
		ch = is.get();
		if (!is) break;
	}
	return res;
}

//  读入一条数据，cnt为属性个数，flg为1表示不读入标签
ml::data parse(istream& is, int cnt, bool flg) {
    ml::data res;
    if (!flg) res.second = stoi(read(is));
	for (int i = 1; i != cnt; ++i)
		res.first.push_back(stof(read(is)));
	return res;
}

//  从文件流读入有cnt个属性，line条数据的数据集
vector<ml::data> read_from_file(int cnt, int line, istream& is, bool test = false) {
	vector<ml::data> v;
	for (int i = test ? 1 : 0; i != cnt; ++i)
		read(is);

	for (int i = 0; i != line; ++i) {
		try {
			ml::data d = parse(is, cnt, test);
            v.push_back(d);
		}
		catch (...) {
			continue;
		}
	}

	return v;
}


int main(void) {

    //typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Mat;
    typedef vector<ml::data> dataset_t;

    const int attr_cnt = 785; //, k = 25;

    ifstream ifstrain("train.csv"), ifstest("test.csv");
    cout << "Loading: ";
    dataset_t dataset = read_from_file(attr_cnt, 42000, ifstrain, false)
                     ,testset = read_from_file(attr_cnt, 28000, ifstest, true)
                     ;
    cout << "Complete!" << endl;

    ml::model m(22);
    m.train(dataset);
    vector<int> res = m.predict(testset);
//
//    ofstream fout("submission_5.csv");
//    int cnt = 0;
//    fout << "ImageId,Label" << endl;
//    for (int t : res)
//        fout << ++cnt << ',' << t << endl;

    vector<int> ks = { 15, 22, 25, 28, 35, 50, 100, 150 };
    for (int k : ks) {
        ml::model m(k);
        cout << "k=" << k << ": ";
        vector<double> res;
        const int test_cnt = 10;
        vector<int> h;
        for (int i = 0; i != test_cnt; ++i) {
            res.push_back(m.evaluate(dataset, 0, 0, 0.8));
            h.push_back(m.get_height());
            cout << res.back() << " \n"[i == test_cnt-1];
        }
        cout << "avg=" << std::setprecision(4) << std::fixed << accumulate(res.begin(), res.end(), 0.) / res.size() <<endl;
        cout << "avgh=" << std::setprecision(4) << std::fixed << accumulate(h.begin(), h.end(), 0.) / h.size() << endl;

    }

//    vector<dataset_t> ds = ml::split_dataset(dataset, { 0.8,  0.2 });
//
//    cout << "Complete!" << endl;
//    ml::PCA pca;
//    pca.build(trainset, k);
//
//    cout << "Transforming: ";
//    trainset = pca.transform(trainset);
//    testset = pca.transform(testset);
//    cout << "Complete!" << endl;
//
//    cout << "Training: " << endl;
//    ml::decision_tree t(k);
//    t.pre_pruning_build(trainset);
//    cout << "Complete!" << endl;
//
	return 0;
}
