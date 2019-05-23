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

    //  分类器接收一条数据，并
    typedef function<int(const data& d)> classifier;

	class decision_tree {
	public:

        //  放在节点上的分类Functor
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

		struct node {
			node() : is_leaf(false) { c[0] = c[1] = 0; }
			node* c[2];
			classifier f;
			int res;
			bool is_leaf;
		};


		int attr_cnt;	//	attribute count
		classifier clf0;	//	main classifier
		node* root;

		void destroy(node*& p) {
			if (!p) return;
			else if (!p->is_leaf) {
                destroy(p->c[0]);
                destroy(p->c[1]);
			}
			delete p;
			p = 0;
		}

		template<class BidIt>
		void count(int* c, BidIt first, BidIt last) {
            fill(c, c + 10, 0);
			for (auto it = first; it != last; ++it)
				c[clf0(*it)]++;
		}

        double cal_ent(int* c, int s) {
            double res = 0;
            for (int i = 0; i != 10; ++i) {
                if (!c[i]) continue;
                double p = (double)c[i] / s;
                res -= p * log(p);
            }
            return res;
        }

        double cal_gain(int* c, int* lc, int* rc) {
            int s = 0, ls = 0, rs = 0;
            for (int i = 0; i != 10; ++i) {
                s += c[i];
                ls += lc[i];
                rs += rc[i];
            }
            double e = cal_ent(c, s),
                   e1 = cal_ent(lc, ls),
                   e0 = cal_ent(rc, rs);
            if (!ls) e1 = 0;
            if (!rs) e0 = 0;
            return e - (ls * e1 + rs * e0) / (ls + rs);
        }

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

		static inline int cal_res(int* c) {
		    return max_element(c, c + 10) - c; //c[0] <= c[1] ? 1 : 0;
        }

		static inline double cal_acc(int* c) {
		    int p = cal_res(c);
            if (c[p] == 0) return 1.0;
            else return (double)(c[p]) / accumulate(c, c + 10, 0);
		}

		static inline bool single_class(int* c) {
            return *max_element(c, c + 10) == accumulate(c, c + 10, 0);
        }

		template<class BidIt>
		bool max_gain_classifier(classifier& res, BidIt first, BidIt last) {
			double max_gain = 0;
			bool flag = 0;	//	set to true when we have a candidate(classifier)
			int c[10]; count(c, first, last);
			for (int i = 0; i != attr_cnt; ++i) {
                sort(first, last, [i](const ml::data& d1, const ml::data& d2) {
                     return d1.first[i] < d2.first[i];
                });

                int lc[10] = {0}, rc[10] = {0}; copy_n(c, 10, rc);
                for (auto it = first; it != prev(last); ++it) {
                    if (it->first[i] == next(it)->first[i]) continue;
                    float v = (it->first[i] + (next(it)->first[i])) / 2;
                    classifier f = attr_clf({ v, i });
                    lc[clf0(*it)]++;
                    rc[clf0(*it)]--;
                    double gain = cal_gain(c, lc, rc);

                    if (gain > max_gain) {
                        max_gain = gain;
                        res = f;
                        flag = 1;
                    }
                }

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

		template<class BidIt>
		void build(node*& p, BidIt first, BidIt last) {
		    #if BUILD_INFO
		    cout << "Build! " << distance(first, last) << endl;
		    #endif
			if (first == last) throw runtime_error("Empty range!");
			int c[10]; count(c, first, last);
			p = new node();
			classifier f;
			if (single_class(c)) {
                #if BUILD_INFO
                cout << "Single class!" << endl;
                #endif
				p->res = clf0(*first);
				p->is_leaf = true;
			}
			else if (!max_gain_classifier(f, first, last)) {
                #if BUILD_INFO
                cout << "Classifier selection failed!" << endl;
                #endif
				p->res = cal_res(c);
				p->is_leaf = true;
			}
			else {
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

		int height(node* p) const {
			if (p->is_leaf) return 1;
			else return max(height(p->c[0]), height(p->c[1])) + 1;
		}

		int max_height() const {
			return height(root);
		}

		int predict(const data& d) const {
			node* p = root;
			if (!p) throw std::runtime_error("Decision tree not built!");
			while (!p->is_leaf)
				p = p->c[p->f(d)];
			return p->res;
		}

	};

    using Eigen::Matrix;
	using Eigen::Dynamic;
    using Eigen::EigenSolver;
    typedef Matrix<double, Dynamic, Dynamic> Mat;
    typedef Matrix<double, Dynamic, 1> Vec;

    class PCA {
    public:
        Mat W, Wt;
        Vec xc;
        int k;

        void build(const dataset_t& trainset, int k_) {
            //  Reduce dimension m to k
            int m = trainset[0].first.size();
            int n = trainset.size();
            k = k_;
            Mat X(m, n);
            for (int i = 0; i != n; ++i)
                for (int j = 0; j != m; ++j)
                    X(j, i) = trainset[i].first[j];

            xc = X * (Vec::Ones(n) / n);
            for (int i = 0; i != n; ++i)
                for (int j = 0; j != m; ++j)
                    X(j, i) -= xc(j);

            Mat tmp = X * X.transpose();
            #if BUILD_INFO
            cout << "Computing Eigenvectors: ";
            #endif
            EigenSolver<Mat> ES(tmp, true);
            #if BUILD_INFO
            cout << "Complete!" << endl;
            #endif

            auto eigenvalues = ES.eigenvalues();
            auto eigenvectors = ES.eigenvectors();
            //for (int i = 0; i != m; ++i) cout << eigenvalues(i) << ' ';
            //cout << string(60, '$') << endl;

            priority_queue<int, vector<int>, function<bool(int, int)>>
            pq([&eigenvalues](int i1, int i2) { return eigenvalues(i1).real() > eigenvalues(i2).real(); });

            for (int i = 0; i != m; ++i) {
                pq.push(i);
                if (pq.size() > k) pq.pop();
            }


            W = Mat(m, k);
            for (int i = 0; i != k; ++i) {
                int t = pq.top(); pq.pop();
                //cout << (long long)eigenvalues(t).real() << " \n"[i == k - 1];
                for (int j = 0; j != m; ++j)
                    W(j, i) = eigenvectors.col(t)(j).real();
            }
            Wt = W.transpose();
        }

        dataset_t transform(const dataset_t& dataset) {
            int m = dataset[0].first.size(), n = dataset.size();
            dataset_t res(n, data(vector<float>(k, 0), 0));
            for (int i = 0; i != dataset.size(); ++i) {
                Vec x(m), xv;
                for (int j = 0; j != m; ++j)
                    x(j) = dataset[i].first[j];
                xv = Wt * x;
                for (int j = 0; j != k; ++j)
                    res[i].first[j] = xv(j);
                res[i].second = dataset[i].second;
            }
            return res;
        }

    };

    vector<dataset_t> split_dataset(dataset_t& dataset, vector<double> split_point) {
        mt19937_64 mt(time(0));
        shuffle(dataset.begin(), dataset.end(), mt);
        vector<dataset_t> res;
        int n = dataset.size();
        double p = 0;
        for (int i = 0; i != split_point.size(); ++i) {
            int l = n * p, r = n * (p + split_point[i]);
            res.push_back(dataset_t(dataset.begin() + l, dataset.begin() + r));
            p += split_point[i];
        }
        int l = n * p, r = n;
        res.push_back(dataset_t(dataset.begin() + l, dataset.begin() + r));
        return res;
    }

    class model {
    public:
        decision_tree tree;
        PCA pca;
        int k;

        model(int k_) : k(k_) {}

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

        vector<int> predict(const dataset_t& testset) {
            dataset_t testset_ = pca.transform(testset);
            vector<int> res;
            for (int i = 0; i != testset_.size(); ++i)
                res.push_back(tree.predict(testset_[i]));
            return res;
        }

        //  returns accuracy on testset
        double evaluate(dataset_t& dataset, int flg, double p = 0.6, double p2 = 0.8) {
            vector<dataset_t> datasets = split_dataset(dataset, { p2 });
            dataset_t& trainset = datasets[0];
            dataset_t& testset = datasets[1];
            train(trainset, flg, p);
            vector<int> result = predict(testset);
            int cnt = 0, tot = testset.size();
            for (int i = 0; i != tot; ++i)
                if (result[i] == testset[i].second)
                    cnt++;
            return (double)cnt / tot;
        }

        int get_height() {
            return tree.max_height();
        }
    };

}

using namespace std;

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

ml::data parse(istream& is, int cnt, bool flg) {
    ml::data res;
    if (!flg) res.second = stoi(read(is));
	for (int i = 1; i != cnt; ++i)
		res.first.push_back(stof(read(is)));
	return res;
}

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
