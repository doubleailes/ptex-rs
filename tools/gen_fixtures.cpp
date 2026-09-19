// Generates .ptx fixture files with the reference C++ PtexWriter and dumps
// expected reader output (header info, face data, reductions, pixels) for
// the Rust port's integration tests.
//
// To regenerate tests/fixtures/:
//   git clone --depth 1 --branch v2.4.3 https://github.com/wdas/ptex.git
//   cmake -S ptex -B ptex/build -DCMAKE_BUILD_TYPE=Release && cmake --build ptex/build -j
//   g++ -O2 -std=c++11 -I ptex/src/ptex tools/gen_fixtures.cpp -o gen_fixtures \
//       -L ptex/build/src/ptex -lPtex -lz -lpthread
//   ./gen_fixtures tests/fixtures
#include <Ptexture.h>
#include <PtexHalf.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
using namespace Ptex;

static void fatal(const std::string& msg) {
    fprintf(stderr, "FATAL: %s\n", msg.c_str());
    exit(1);
}

// deterministic data generators
static uint8_t u8val(int f, int c, int u, int v) {
    return (uint8_t)((u * 3 + v * 7 + c * 17 + f * 29) & 0xff);
}
static uint16_t u16val(int f, int c, int u, int v) {
    return (uint16_t)((u * 313 + v * 771 + c * 1717 + f * 2900) & 0xffff);
}
static float f32val(int f, int c, int u, int v) {
    return 0.01f * (float)u + 0.002f * (float)v + 1.5f * (float)c + 10.0f * (float)f;
}

static void writeQuadU8(const char* path) {
    Ptex::String err;
    const int nfaces = 4;
    PtexPtr<PtexWriter> w(PtexWriter::open(path, mt_quad, dt_uint8, 4, 3, nfaces, err));
    if (!w) fatal(err.c_str());

    Res res[nfaces] = { Res(4, 4), Res(3, 2), Res(5, 5), Res(2, 3) };
    int adjfaces[nfaces][4] = {
        {3, 1, -1, 2}, {0, 2, -1, -1}, {1, 3, 0, -1}, {2, 0, -1, 1},
    };
    int adjedges[nfaces][4] = {
        {2, 3, 0, 1}, {1, 3, 0, 0}, {2, 0, 0, 1}, {2, 3, 0, 1},
    };

    for (int f = 0; f < nfaces; f++) {
        FaceInfo fi(res[f], adjfaces[f], adjedges[f]);
        int uw = res[f].u(), vw = res[f].v();
        if (f == 2) {
            // deliberately constant face
            uint8_t pixel[4] = { 40, 80, 120, 200 };
            if (!w->writeConstantFace(f, fi, pixel)) fatal("writeConstantFace");
        } else {
            std::vector<uint8_t> data(uw * vw * 4);
            for (int v = 0; v < vw; v++)
                for (int u = 0; u < uw; u++)
                    for (int c = 0; c < 4; c++)
                        data[(v * uw + u) * 4 + c] = u8val(f, c, u, v);
            if (!w->writeFace(f, fi, &data[0])) fatal("writeFace");
        }
    }

    // meta data of every type, plus a large entry (>1024 bytes -> LMD block)
    w->writeMeta("sval", "a string value");
    int8_t i8vals[3] = { -1, 2, -3 };
    w->writeMeta("i8vals", i8vals, 3);
    int16_t i16vals[4] = { -1000, 2000, -3000, 4000 };
    w->writeMeta("i16vals", i16vals, 4);
    int32_t i32vals[2] = { -100000, 200000 };
    w->writeMeta("i32vals", i32vals, 2);
    float fvals[3] = { 0.5f, -1.25f, 3.75f };
    w->writeMeta("fvals", fvals, 3);
    double dvals[2] = { 1.0e10, -2.5 };
    w->writeMeta("dvals", dvals, 2);
    std::vector<double> big(300);
    for (int i = 0; i < 300; i++) big[i] = 0.25 * i;
    w->writeMeta("bigvals", &big[0], 300);

    if (!w->close(err)) fatal(err.c_str());
}

static void writeQuadF32(const char* path) {
    Ptex::String err;
    const int nfaces = 2;
    PtexPtr<PtexWriter> w(PtexWriter::open(path, mt_quad, dt_float, 3, -1, nfaces, err));
    if (!w) fatal(err.c_str());

    Res res[nfaces] = { Res(7, 7), Res(5, 5) }; // 128x128x12B face (196KB) gets split into 2 tiles
    for (int f = 0; f < nfaces; f++) {
        FaceInfo fi(res[f]);
        int uw = res[f].u(), vw = res[f].v();
        std::vector<float> data((size_t)uw * vw * 3);
        for (int v = 0; v < vw; v++)
            for (int u = 0; u < uw; u++)
                for (int c = 0; c < 3; c++)
                    data[((size_t)v * uw + u) * 3 + c] = f32val(f, c, u, v);
        if (!w->writeFace(f, fi, &data[0])) fatal("writeFace");
    }
    if (!w->close(err)) fatal(err.c_str());
}

static void writeQuadF16(const char* path) {
    Ptex::String err;
    const int nfaces = 2;
    PtexPtr<PtexWriter> w(PtexWriter::open(path, mt_quad, dt_half, 3, -1, nfaces, err));
    if (!w) fatal(err.c_str());

    Res res[nfaces] = { Res(5, 4), Res(4, 5) };
    for (int f = 0; f < nfaces; f++) {
        FaceInfo fi(res[f]);
        int uw = res[f].u(), vw = res[f].v();
        std::vector<PtexHalf> data((size_t)uw * vw * 3);
        for (int v = 0; v < vw; v++)
            for (int u = 0; u < uw; u++)
                for (int c = 0; c < 3; c++)
                    data[((size_t)v * uw + u) * 3 + c] = 0.125f * f32val(f, c, u, v);
        if (!w->writeFace(f, fi, &data[0])) fatal("writeFace");
    }
    if (!w->close(err)) fatal(err.c_str());
}

static void writeTriU16(const char* path) {
    Ptex::String err;
    const int nfaces = 4;
    PtexPtr<PtexWriter> w(PtexWriter::open(path, mt_triangle, dt_uint16, 1, -1, nfaces, err));
    if (!w) fatal(err.c_str());

    Res res[nfaces] = { Res(5, 5), Res(4, 4), Res(6, 6), Res(2, 2) };
    for (int f = 0; f < nfaces; f++) {
        FaceInfo fi(res[f]);
        int uw = res[f].u(), vw = res[f].v();
        std::vector<uint16_t> data((size_t)uw * vw);
        for (int v = 0; v < vw; v++)
            for (int u = 0; u < uw; u++)
                data[(size_t)v * uw + u] = u16val(f, 0, u, v);
        if (!w->writeFace(f, fi, &data[0])) fatal("writeFace");
    }
    if (!w->close(err)) fatal(err.c_str());
}

// A single-channel uint8 face big enough that the writer tiles it in both
// directions, and whose first mipmap level is tiled too.
//
// PtexWriter tiles a face whose uncompressed data exceeds TileSize (64 KB).
// 1024x512x1B = 512 KB gives 8 tiles in a 4x2 grid at level 0, and the
// 512x256 = 128 KB reduction level gives 2 more in a 2x1 grid - neither of
// which the other fixtures exercise (quad_f32 has a single 1x2 grid and no
// tiled reduction).
static void writeQuadTiled(const char* path) {
    Ptex::String err;
    const int nfaces = 2;
    PtexPtr<PtexWriter> w(PtexWriter::open(path, mt_quad, dt_uint8, 1, -1, nfaces, err));
    if (!w) fatal(err.c_str());

    Res res[nfaces] = { Res(10, 9), Res(3, 3) };
    for (int f = 0; f < nfaces; f++) {
        FaceInfo fi(res[f]);
        int uw = res[f].u(), vw = res[f].v();
        std::vector<uint8_t> data((size_t)uw * vw);
        for (int v = 0; v < vw; v++)
            for (int u = 0; u < uw; u++)
                data[(size_t)v * uw + u] = u8val(f, 0, u, v);
        if (!w->writeFace(f, fi, &data[0])) fatal("writeFace");
    }
    if (!w->close(err)) fatal(err.c_str());
}

static void dumpMeta(FILE* out, PtexMetaData* meta) {
    for (int i = 0; i < meta->numKeys(); i++) {
        const char* key;
        MetaDataType type;
        meta->getKey(i, key, type);
        fprintf(out, "meta %s %s", key, MetaDataTypeName(type));
        switch (type) {
        case mdt_string: {
            const char* val;
            meta->getValue(key, val);
            fprintf(out, " %s", val);
            break;
        }
        case mdt_int8: {
            const int8_t* val; int count;
            meta->getValue(key, val, count);
            for (int j = 0; j < count; j++) fprintf(out, " %d", (int)val[j]);
            break;
        }
        case mdt_int16: {
            const int16_t* val; int count;
            meta->getValue(key, val, count);
            for (int j = 0; j < count; j++) fprintf(out, " %d", (int)val[j]);
            break;
        }
        case mdt_int32: {
            const int32_t* val; int count;
            meta->getValue(key, val, count);
            for (int j = 0; j < count; j++) fprintf(out, " %d", val[j]);
            break;
        }
        case mdt_float: {
            const float* val; int count;
            meta->getValue(key, val, count);
            for (int j = 0; j < count; j++) fprintf(out, " %.9g", val[j]);
            break;
        }
        case mdt_double: {
            const double* val; int count;
            meta->getValue(key, val, count);
            for (int j = 0; j < count; j++) fprintf(out, " %.17g", val[j]);
            break;
        }
        }
        fprintf(out, "\n");
    }
}

static void dumpInfo(PtexTexture* tx, const char* base) {
    std::string infopath = std::string(base) + ".info.txt";
    FILE* info = fopen(infopath.c_str(), "w");
    if (!info) fatal("open info");
    fprintf(info, "meshtype %s\n", MeshTypeName(tx->meshType()));
    fprintf(info, "datatype %s\n", DataTypeName(tx->dataType()));
    fprintf(info, "alphachan %d\n", tx->alphaChannel());
    fprintf(info, "nchannels %d\n", tx->numChannels());
    fprintf(info, "nfaces %d\n", tx->numFaces());
    fprintf(info, "hasmipmaps %d\n", tx->hasMipMaps() ? 1 : 0);
    for (int f = 0; f < tx->numFaces(); f++) {
        const FaceInfo& fi = tx->getFaceInfo(f);
        fprintf(info, "face %d res %d %d adjfaces %d %d %d %d adjedges %d %d %d %d flags %d\n",
                f, (int)fi.res.ulog2, (int)fi.res.vlog2,
                fi.adjfaces[0], fi.adjfaces[1], fi.adjfaces[2], fi.adjfaces[3],
                (int)fi.adjedge(0), (int)fi.adjedge(1), (int)fi.adjedge(2), (int)fi.adjedge(3),
                (int)fi.flags);
    }
    PtexPtr<PtexMetaData> meta(tx->getMetaData());
    if (meta) dumpMeta(info, meta);
    fclose(info);
}

static void dump(const char* path, const char* base) {
    Ptex::String err;
    PtexPtr<PtexTexture> tx(PtexTexture::open(path, err, /*premultiply*/ false));
    if (!tx) fatal(err.c_str());
    dumpInfo(tx, base);

    int psize = tx->numChannels() * DataSize(tx->dataType());

    // full-res face data
    std::string facespath = std::string(base) + ".faces.dat";
    FILE* faces = fopen(facespath.c_str(), "wb");
    if (!faces) fatal("open faces");
    for (int f = 0; f < tx->numFaces(); f++) {
        const FaceInfo& fi = tx->getFaceInfo(f);
        std::vector<char> buf((size_t)fi.res.size() * psize);
        tx->getData(f, &buf[0], 0);
        fwrite(&buf[0], 1, buf.size(), faces);
    }
    fclose(faces);

    // reduced-res face data: one symmetric reduction (usually a stored
    // mipmap level) and one asymmetric (dynamically generated), except for
    // triangle meshes where only symmetric reductions are supported
    std::string redpath = std::string(base) + ".reduced.dat";
    FILE* red = fopen(redpath.c_str(), "wb");
    if (!red) fatal("open reduced");
    bool tri = tx->meshType() == mt_triangle;
    for (int f = 0; f < tx->numFaces(); f++) {
        const FaceInfo& fi = tx->getFaceInfo(f);
        Res r1((int8_t)std::max(0, fi.res.ulog2 - 1), (int8_t)std::max(0, fi.res.vlog2 - 1));
        {
            std::vector<char> buf((size_t)r1.size() * psize);
            tx->getData(f, &buf[0], 0, r1);
            fwrite(&buf[0], 1, buf.size(), red);
        }
        Res r2 = tri
            ? Res((int8_t)std::max(0, fi.res.ulog2 - 2), (int8_t)std::max(0, fi.res.vlog2 - 2))
            : Res((int8_t)std::max(0, fi.res.ulog2 - 2), (int8_t)std::max(0, fi.res.vlog2 - 1));
        {
            std::vector<char> buf((size_t)r2.size() * psize);
            tx->getData(f, &buf[0], 0, r2);
            fwrite(&buf[0], 1, buf.size(), red);
        }
    }
    fclose(red);

    // premultiplied full-res face data (only for files with alpha)
    if (tx->alphaChannel() >= 0) {
        PtexPtr<PtexTexture> ptx(PtexTexture::open(path, err, /*premultiply*/ true));
        if (!ptx) fatal(err.c_str());
        std::string pmpath = std::string(base) + ".pmfaces.dat";
        FILE* pm = fopen(pmpath.c_str(), "wb");
        if (!pm) fatal("open pmfaces");
        for (int f = 0; f < ptx->numFaces(); f++) {
            const FaceInfo& fi = ptx->getFaceInfo(f);
            std::vector<char> buf((size_t)fi.res.size() * psize);
            ptx->getData(f, &buf[0], 0);
            fwrite(&buf[0], 1, buf.size(), pm);
        }
        fclose(pm);
    }

    // pixel samples
    std::string pixpath = std::string(base) + ".pixels.txt";
    FILE* pix = fopen(pixpath.c_str(), "w");
    if (!pix) fatal("open pixels");
    int nchan = tx->numChannels();
    std::vector<float> result(nchan);
    for (int f = 0; f < tx->numFaces(); f++) {
        const FaceInfo& fi = tx->getFaceInfo(f);
        int uw = fi.res.u(), vw = fi.res.v();
        int us[3] = { 0, uw / 2, uw - 1 };
        int vs[3] = { 0, vw / 2, vw - 1 };
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                tx->getPixel(f, us[i], vs[j], &result[0], 0, nchan);
                fprintf(pix, "%d %d %d", f, us[i], vs[j]);
                for (int c = 0; c < nchan; c++) fprintf(pix, " %.9g", result[c]);
                fprintf(pix, "\n");
            }
        }
    }
    fclose(pix);
}

// The tiled fixture's face data is half a megabyte, too big to commit, so
// only the header info and a dense grid of pixel samples - many inside every
// tile - are recorded.  The Rust tests verify the tile path by reassembling
// tiles and comparing against the whole-face read, and verify the decode
// itself against these samples.
static void dumpTiled(const char* path, const char* base) {
    Ptex::String err;
    PtexPtr<PtexTexture> tx(PtexTexture::open(path, err, /*premultiply*/ false));
    if (!tx) fatal(err.c_str());
    dumpInfo(tx, base);

    std::string pixpath = std::string(base) + ".pixels.txt";
    FILE* pix = fopen(pixpath.c_str(), "w");
    if (!pix) fatal("open pixels");
    int nchan = tx->numChannels();
    std::vector<float> result(nchan);
    for (int f = 0; f < tx->numFaces(); f++) {
        const FaceInfo& fi = tx->getFaceInfo(f);
        int uw = fi.res.u(), vw = fi.res.v();
        int ustep = uw > 16 ? uw / 16 : 1;
        int vstep = vw > 16 ? vw / 16 : 1;
        for (int v = 0; v < vw; v += vstep) {
            for (int u = 0; u < uw; u += ustep) {
                tx->getPixel(f, u, v, &result[0], 0, nchan);
                fprintf(pix, "%d %d %d", f, u, v);
                for (int c = 0; c < nchan; c++) fprintf(pix, " %.9g", result[c]);
                fprintf(pix, "\n");
            }
        }
    }
    fclose(pix);
}

int main(int argc, char** argv) {
    if (argc != 2) fatal("usage: gen_fixtures <outdir>");
    std::string dir = argv[1];

    struct {
        const char* name;
        void (*write)(const char*);
        void (*dump)(const char*, const char*);
    } fixtures[] = {
        { "quad_u8", writeQuadU8, dump },
        { "quad_f32", writeQuadF32, dump },
        { "quad_f16", writeQuadF16, dump },
        { "tri_u16", writeTriU16, dump },
        { "quad_tiled", writeQuadTiled, dumpTiled },
    };
    for (auto& fx : fixtures) {
        std::string ptx = dir + "/" + fx.name + ".ptx";
        std::string base = dir + "/" + fx.name;
        fx.write(ptx.c_str());
        fx.dump(ptx.c_str(), base.c_str());
        printf("wrote %s\n", ptx.c_str());
    }
    return 0;
}
