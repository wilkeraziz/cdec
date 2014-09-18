#ifndef _STATELESS_KLM_FF_H_
#define _STATELESS_KLM_FF_H_
//TODO StatelessKLanguageModel and KLanguageModel share a lot of code, thus I should get rid of duplicates (wilker)


#include <vector>
#include <string>

#include "ff_factory.h"
#include "ff.h"

template <class Model> struct StatelessKLanguageModelImpl;
template <class Model>
class StatelessKLanguageModel : public FeatureFunction {
public:
    StatelessKLanguageModel(const std::string& param);
    ~StatelessKLanguageModel();
    static std::string usage(bool param,bool verbose);
protected:
    virtual void TraversalFeaturesImpl(const SentenceMetadata& smeta,
            const HG::Edge& edge,
            const std::vector<const void*>& ant_contexts,
            SparseVector<double>* features,
            SparseVector<double>* estimated_features,
            void* out_context) const;
private:
    int fid_;        // LanguageModel
    int oov_fid_;    // LanguageModel_OOV
    int emit_fid_;   // LanguageModel_Emit [only used for class-based LMs]
    StatelessKLanguageModelImpl<Model>* pimpl_;
};

struct StatelessKLanguageModelFactory : public FactoryBase<FeatureFunction> {
    FP Create(std::string param) const;
    std::string usage(bool params,bool verbose) const;
};

#endif // _STATELESS_KLM_FF_H_
